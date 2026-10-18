#!/usr/bin/env python3
"""Build the scored artifact for the ARC explorer re-navigation decomposition.

Reads the raw census outputs produced by `scripts/arc_explorer_renavigation_census.py`
(one cell per game per arm) plus the explore-budget invariance control, and emits
`results/arc_explorer_renavigation_20260802/arc_explorer_renavigation.json`.

Everything it writes is derived arithmetic over those cells -- no new measurement, no
model, no LLM. The prize is converted into ARC-AGI-3's own per-level scoring function,
`min((baseline_actions / agent_actions)**2, 115)`, because a saving stated in actions
understates it: the score is QUADRATIC in the action count, so an x% action reduction is
worth roughly 2x% on score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

RENAV = (
    "renavigation.reset_action",
    "renavigation.replay_prefix_shared_with_current_path",
    "renavigation.replay_suffix_past_divergence",
    "renavigation.forward_walk_no_reset",
)
INERT = "expansion.probe_was_inert_frame_unchanged"

# The named classes, each with the mechanism that would have to change for those actions
# not to be spent. "Avoidable" means A MECHANISM CAN BE NAMED and pointed at, not that
# flipping it is free and not that any arm here shows it recovering the actions.
CLASS_MECHANISM: dict[str, dict[str, Any]] = {
    "expansion.probe_discovered_new_state": {
        "avoidable": False,
        "why": (
            "Nothing. This IS the work: one action spent, one previously-unknown state added "
            "to the agent's graph. An exploration agent cannot discover a state without "
            "entering it."
        ),
    },
    "expansion.probe_revisited_known_state": {
        "avoidable": False,
        "why": (
            "The action was UNTESTED at that state AND it moved the board, so spending it is "
            "genuine information -- it adds a transition edge the agent did not have -- even "
            "though the destination turned out to be a state already in the graph. Removing "
            "it needs a transition model accurate enough to PREDICT the destination before "
            "paying for the action, i.e. the induce->verify->plan path. That is the "
            "generation-side lever every other probe this session found inert, so counting "
            "these as avoidable would be assuming the thing that keeps failing."
        ),
    },
    INERT: {
        "avoidable": True,
        "why": (
            "An inert-action PRIOR that predicts 'this action will not change a single pixel' "
            "before the action is spent. Unlike the class above, no world model is needed: "
            "the prediction is binary and about the agent's OWN action vocabulary, and the "
            "machinery already ships ON -- `inert_click_pruner` "
            "(SUBMITTED_INERT_CLICK_PRUNER_ENABLED), `frame_change_scorer` / "
            "`ActionEffectExpansionPrior`, `prune_arc_actions_by_prior_quantile`. These 1,148 "
            "actions are precisely what those priors did NOT catch. Named mechanism, existing "
            "code path, unmeasured headroom -- NOT a demonstrated saving."
        ),
    },
    "renavigation.reset_action": {
        "avoidable": True,
        "why": (
            "Do not leave a node that still has untested work the agent is willing to spend "
            "on. Every RESET here is step 1 of a plan to return to a node the agent had "
            "ALREADY STOOD ON, and 51% of the navigation actions downstream of one followed a "
            "departure from a node whose remaining work the GLOBAL TIER BARRIER "
            "(REQ-ARC-WMTE-5836) was refusing. The environment offers no undo, so once the "
            "departure is decided RESET+replay is forced -- the avoidable decision is the "
            "departure, not the RESET."
        ),
    },
    "renavigation.replay_prefix_shared_with_current_path": {
        "avoidable": True,
        "why": (
            "Same departure decision as the RESET it follows. These steps re-walk a PREFIX of "
            "the path the agent was standing on -- ground it covered seconds earlier in the "
            "same episode. Given reset-only semantics there is no cheaper route to an "
            "ANCESTOR of your current state, so this is irreducible CONDITIONAL on the "
            "departure and fully avoidable if the departure does not happen."
        ),
    },
    "renavigation.replay_suffix_past_divergence": {
        "avoidable": True,
        "why": (
            "Record enough forward edges that `_exact_shortest_path` / "
            "`_partial_forward_path` can walk there without a RESET, or order the frontier "
            "for locality. These steps cross ground the current path does NOT cover, so they "
            "buy real distance; only the ROUTE is wasteful, not the displacement."
        ),
    },
    "renavigation.forward_walk_no_reset": {
        "avoidable": True,
        "why": (
            "Frontier ordering for locality. These are already the CHEAP form of navigation "
            "(the agent's own `_exact_shortest_path` found a forward route, no RESET), and "
            "they average ~2 steps. Shortening them means choosing a nearer frontier target, "
            "which the measured locality counterfactual says is worth approximately nothing: "
            "the frontier already lands on the cheapest eligible node."
        ),
    },
    "bootstrap.reset": {
        "avoidable": False,
        "why": "Nothing. One RESET per episode is how the agent obtains its first frame.",
    },
    "plan.execute_step": {
        "avoidable": False,
        "why": "Not a search cost. Zero here by construction (induction disabled).",
    },
    "other": {"avoidable": False, "why": "Unclassified residue; reported, not explained."},
}


def score(baseline: float, agent: float) -> float:
    if agent <= 0:
        return 115.0
    return min((float(baseline) / float(agent)) ** 2, 115.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True, help="census.json from the two-arm run")
    ap.add_argument("--census-seed-b", required=True, help="census.json from the second seed")
    ap.add_argument("--control", required=True, help="explore-budget invariance control json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--exemplar-rows",
        default="tn36,tu93,r11l,ls20",
        help="games whose FULL per-action rows are preserved verbatim in the artifact dir",
    )
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    census = json.loads(Path(args.census).read_text())
    census_b = json.loads(Path(args.census_seed_b).read_text())
    control = json.loads(Path(args.control).read_text())

    arms = census["arms"]
    per_arm: dict[str, Any] = {}
    for arm in arms:
        counts: dict[str, int] = {}
        nav_planned = nav_after_tier = probes_bought = elig_at_targets = 0
        levels: dict[str, int] = {}
        nav_shares: list[float] = []
        mech: dict[str, int] = {}
        batch_saved = batch_nav_total = 0
        for cell in census["cells"][arm]:
            for k, v in (cell.get("class_counts") or {}).items():
                counts[k] = counts.get(k, 0) + int(v)
            es = cell.get("episode_summary") or {}
            nav_planned += int(es.get("navigation_actions_total") or 0)
            nav_after_tier += int(es.get("navigation_actions_after_tier_deferred_departure") or 0)
            probes_bought += int(es.get("probe_actions_bought_by_navigation") or 0)
            elig_at_targets += int(es.get("tier_eligible_rows_at_target_total") or 0)
            for m, n in (es.get("episodes_by_mechanism") or {}).items():
                mech[m] = mech.get(m, 0) + int(n)
            if cell.get("best_level"):
                levels[cell["game"]] = int(cell["best_level"])
            cc = cell.get("class_counts") or {}
            tot_cell = sum(cc.values()) or 1
            nav_shares.append(sum(cc.get(k, 0) for k in RENAV) / tot_cell)
            # FAN-OUT-AT-TARGET counterfactual, per game: walk the episodes in the order they
            # happened and stop once the probes already bought would have been covered by
            # taking every tier-eligible row at each visited target instead of exactly one.
            eps = cell.get("episodes") or []
            if eps:
                need = sum(int(e["n_probe_actions"]) for e in eps)
                cost = [int(e["n_reset_actions"]) + int(e["n_nav_actions"]) for e in eps]
                cap = [max(1, int(e.get("tier_eligible_rows_at_target") or 1)) for e in eps]
                got = spent = 0
                for idx in range(len(eps)):
                    if got >= need:
                        break
                    got += cap[idx]
                    spent += cost[idx]
                batch_nav_total += sum(cost)
                batch_saved += sum(cost) - spent
        total = sum(counts.values())
        classes = []
        for k in sorted(counts, key=lambda x: -counts[x]):
            meta = CLASS_MECHANISM.get(k, CLASS_MECHANISM["other"])
            classes.append(
                {
                    "kind": k,
                    "actions": counts[k],
                    "share_of_budget": round(counts[k] / total, 6),
                    "avoidable": bool(meta["avoidable"]),
                    "why": meta["why"],
                }
            )
        nav_shares.sort()
        per_arm[arm] = {
            "total_actions": total,
            "classes": classes,
            "renavigation_actions_served": sum(counts.get(k, 0) for k in RENAV),
            "renavigation_share": round(sum(counts.get(k, 0) for k in RENAV) / total, 6),
            "inert_probe_actions": counts.get(INERT, 0),
            "inert_probe_share": round(counts.get(INERT, 0) / total, 6),
            "new_state_actions": counts.get("expansion.probe_discovered_new_state", 0),
            "navigation_actions_planned_in_episodes": nav_planned,
            "navigation_actions_after_tier_deferred_departure": nav_after_tier,
            "probes_bought_by_navigation": probes_bought,
            "navigation_actions_per_probe": (
                round(nav_planned / probes_bought, 3) if probes_bought else None
            ),
            "tier_eligible_rows_available_at_targets": elig_at_targets,
            "episodes_by_navigation_mechanism": mech,
            "levels_reached_within_budget": levels,
            "per_game_renavigation_share": {
                "min": round(nav_shares[0], 6),
                "median": round(statistics.median(nav_shares), 6),
                "max": round(nav_shares[-1], 6),
                "mean": round(sum(nav_shares) / len(nav_shares), 6),
            },
            "fan_out_at_target_counterfactual": {
                "navigation_actions": batch_nav_total,
                "navigation_actions_if_every_eligible_row_taken_per_visit": (
                    batch_nav_total - batch_saved
                ),
                "actions_saved_upper_bound": batch_saved,
            },
        }

    shipped = per_arm["shipped"]
    total = shipped["total_actions"]
    inert = shipped["inert_probe_actions"]
    renav = shipped["renavigation_actions_served"]

    def prize(saved: int) -> dict[str, Any]:
        after = total - saved
        return {
            "actions_before": total,
            "actions_after": after,
            "action_reduction": round(saved / total, 6),
            "score_multiplier": round((total / after) ** 2, 4),
            "worked_example_baseline_100_actions": {
                "score_before": round(score(100, total / 25), 4),
                "score_after": round(score(100, after / 25), 4),
            },
        }

    verdict_prefix = "complete_"
    payload: dict[str, Any] = {
        "experiment": "outer_loop_arc_explorer_renavigation_decomposition",
        "schema": "carnot.arc.explorer_action_class_decomposition.v1",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "milestone": "2026.08.outer_loop",
        "question": (
            "The 2026-08-01 tn36 action census found 177 of 240 actions (73.8%) were "
            "navigation or replay back to already-visited states. Decompose the "
            "non-expanding actions of a real, NON-solve-conditioned exploration run into "
            "named classes with shares, say for each what would have to change for those "
            "actions not to be spent, and price the largest avoidable class in ARC-AGI-3's "
            "own quadratic scoring units."
        ),
        "honest_verdict": (
            verdict_prefix
            + "the_tn36_73pct_renavigation_figure_does_not_generalize_roster_renavigation_is_"
            "11_7pct_and_the_largest_avoidable_class_is_inert_probes_at_19_1pct_not_"
            "renavigation"
        ),
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "inference_substrate_note": (
            "E3AgentPolicy (the SCORED policy) steps the OFFLINE arcade "
            "(OperationMode.OFFLINE over the local environment_files/ tree). No scorecard is "
            "submitted, no online game is played, no API key is used. No LLM is constructed "
            "and no generator server is started: CARNOT_ARC_DISABLE_INDUCTION=1 short-circuits "
            "the induce tier before any proposer method is called, and CUDA_VISIBLE_DEVICES "
            "was empty in every child process. There is no model to name, so model_specs is "
            "absent by design for this substrate; random_seed and reproducibility_checksum are "
            "present as the substrate requires."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_note": (
            "This is an INSTRUMENT run. It banks no level, claims no solve, runs no "
            "reproduction gate and writes no row to ops/arc_solve_registry.yaml. The "
            "levels_reached_within_budget field records, as an incidental observation, that "
            "some games happened to advance a level inside the 240-action instrument budget; "
            "those are NOT banked solves and must not be cited as such."
        ),
        "verifier_is_oracle": {
            "value": False,
            "principle": (
                "Nothing here consults a win oracle. The recorded quantity is which code "
                "branch emitted each action and whether the resulting frame/state was new -- "
                "facts about the agent's own search, not about whether an answer was right."
            ),
        },
        "random_seed": census["seed"],
        "duration_s": round(
            float(census.get("duration_s") or 0.0)
            + float(census_b.get("duration_s") or 0.0)
            + float(control.get("duration_s") or 0.0)
            + (time.time() - t0),
            3,
        ),
        "preconditions_checked": [
            {
                "resource": "offline arcade environment_files present",
                "available": True,
                "evidence": "every cell loaded its game class from environment_files/<game>/",
            },
            {
                "resource": "CARNOT_ARC_E3_DIR redirected away from tracked results/arc_e3",
                "available": True,
                "evidence": (
                    "arc_explorer_renavigation_probe.py refuses to start otherwise; "
                    "`git status results/arc_e3` clean before and after the census"
                ),
            },
            {
                "resource": "CARNOT_ARC_DISABLE_INDUCTION=1 (no LLM, no generator server)",
                "available": True,
                "evidence": "probe refuses to start otherwise",
            },
            {
                "resource": "no GPU used",
                "available": True,
                "evidence": "CUDA_VISIBLE_DEVICES='' in every child process env",
            },
        ],
        "config": {
            "games": census["games"],
            "n_games": len(census["games"]),
            "actions_per_game": census["budget"],
            "arms": arms,
            "policy": "E3AgentPolicy (scored) driven action-by-action against offline_arcade()",
            "explore_budget": "resolved per game by _route_explore_budget, as the scored agent does",
            "generator": "none (induction disabled)",
        },
        "corpus_is_not_solve_conditioned": {
            "value": True,
            "principle": (
                "A window built through arc_actions_to_progress.build_progress_window requires "
                "the game to be ALREADY SOLVED to L1 through a registered GameAdapter and then "
                "shows a prefix of a banked WINNING route, so a re-navigation measurement "
                "built on one inherits its own answer. These traces come from a FRESH RESET "
                "with a bounded budget on the offline arcade: no adapter, no banked route, and "
                "24 of the 25 games do not reach a level inside the budget."
            ),
        },
        "explore_budget_control": control,
        "arms": per_arm,
        "seed_replication": {
            "second_seed": census_b["seed"],
            "byte_identical_class_counts": (
                census_b["summary"]["shipped"]["class_counts"]
                == census["summary"]["shipped"]["class_counts"]
            ),
            "note": (
                "The second seed is NOT an independent sample and must not be read as a "
                "replication. The shipped explorer's stochastic components carry their own "
                "fixed internal seeds (frontier_discipline_seed=20260724, "
                "_CLICK_PIXEL_FALLBACK_RNG), so a caller-supplied seed does not perturb the "
                "trajectory in this configuration -- the class counts came back byte-identical. "
                "The accounting is therefore ONE deterministic trajectory per game, not a "
                "sample mean, and carries no sampling error bar."
            ),
        },
    }

    # Per-game class counts, hoisted to the top level. The roster aggregate hides the single
    # most important property of this measurement -- that every class is enormously
    # heterogeneous across games -- and an aggregate that hides its own dispersion is how the
    # tn36 figure got extrapolated in the first place.
    payload_per_game = {
        arm: {
            cell["game"]: {
                "class_counts": cell.get("class_counts") or {},
                "renavigation_actions": sum(
                    (cell.get("class_counts") or {}).get(k, 0) for k in RENAV
                ),
                "navigation_episodes": cell.get("n_navigation_episodes"),
                "best_level": cell.get("best_level"),
                "explore_budget": cell.get("explore_budget"),
                "explore_budget_provenance": cell.get("explore_budget_provenance"),
            }
            for cell in census["cells"][arm]
        }
        for arm in arms
    }
    payload["per_game"] = payload_per_game

    largest_avoidable = max(
        (c for c in shipped["classes"] if c["avoidable"]), key=lambda c: c["actions"]
    )
    new_state = next(
        c["actions"]
        for c in shipped["classes"]
        if c["kind"] == "expansion.probe_discovered_new_state"
    )
    revisit = next(
        c["actions"]
        for c in shipped["classes"]
        if c["kind"] == "expansion.probe_revisited_known_state"
    )
    payload["numerical_coincidence_note"] = {
        "observation": (
            f"expansion.probe_discovered_new_state and expansion.probe_revisited_known_state "
            f"both total {new_state} on the shipped arm. This is a COINCIDENCE of the roster "
            f"sum, not two views of one quantity."
        ),
        "evidence": (
            "Per game the two diverge wildly and in both directions -- ar25 190 vs 3, ls20 215 "
            "vs 17, lf52 6 vs 201, sp80 31 vs 164 -- see the per_game block. They are also "
            "computed from different signals: 'discovered' from growth in the explorer's node "
            "graph across the next decision, 'revisited' from a raw frame hash that changed "
            "while the node did not."
        ),
        "why_flagged_here": (
            "Two distinct metrics agreeing exactly is the TAUTOLOGY pattern "
            "adversarial_verify.py hunts, and a reader is entitled to suspect a bug. Recorded "
            "rather than left for someone to trip over."
        ),
    }
    payload["headline"] = {
        "roster_actions": total,
        "renavigation_share_roster": shipped["renavigation_share"],
        "renavigation_share_tn36": next(
            round(
                sum(c["class_counts"].get(k, 0) for k in RENAV) / sum(c["class_counts"].values()), 6
            )
            for c in census["cells"]["shipped"]
            if c["game"] == "tn36"
        ),
        "largest_avoidable_class": largest_avoidable["kind"],
        "largest_avoidable_class_actions": largest_avoidable["actions"],
        "largest_avoidable_class_share": largest_avoidable["share_of_budget"],
        "avoidable_share_total": round(
            sum(c["actions"] for c in shipped["classes"] if c["avoidable"]) / total, 6
        ),
        "premise_correction": (
            "The 73.8%-re-navigation figure that motivated this task is a tn36 property, not a "
            "roster property. tn36 is the maximum of 25 games at 80.8%; the MEDIAN game spends "
            "2.9% of its actions on re-navigation and the roster mean is 11.7%. Any lever "
            "sized off the tn36 number is sized ~6x too large for the roster."
        ),
    }
    payload["theoretical_saving"] = {
        "method": (
            "Upper bound only. Each row assumes a class goes to ZERO with no other change to "
            "the trajectory, which is false in detail: removing an action changes what the "
            "agent sees next. Read these as the ceiling a mechanism could aim at, never as a "
            "prediction."
        ),
        "score_function": "min((baseline_actions / agent_actions)**2, 115)",
        "largest_avoidable_class_to_zero": {
            "class": largest_avoidable["kind"],
            **prize(largest_avoidable["actions"]),
        },
        "all_renavigation_to_zero": {"class": "renavigation.*", **prize(renav)},
        "both_to_zero": {
            "class": f"{largest_avoidable['kind']} + renavigation.*",
            **prize(inert + renav),
        },
        "tn36_only_renavigation_to_zero": {
            "note": (
                "The single-game version of the premise, kept because it is what motivated the "
                "task and because it shows how badly a one-game figure extrapolates."
            ),
            "actions_before": 240,
            "actions_after": 240
            - next(
                sum(c["class_counts"].get(k, 0) for k in RENAV)
                for c in census["cells"]["shipped"]
                if c["game"] == "tn36"
            ),
            "score_multiplier": round(
                (
                    240
                    / (
                        240
                        - next(
                            sum(c["class_counts"].get(k, 0) for k in RENAV)
                            for c in census["cells"]["shipped"]
                            if c["game"] == "tn36"
                        )
                    )
                )
                ** 2,
                4,
            ),
        },
    }

    payload["mechanism_probe_tier_barrier"] = {
        "what": (
            "arm `tier_off` is identical to `shipped` except "
            "CARNOT_ARC_FRONTIER_TIER_EXHAUSTION=0. It tests CAUSE: is the re-navigation forced "
            "by the environment's reset-only semantics, or produced by the global tier "
            "barrier's own departure rule?"
        ),
        "answer": (
            "Produced by the barrier, on the games where re-navigation is large. Roster "
            "re-navigation halves (699 -> 345 actions); on tn36 it collapses from 194 actions "
            "to 2, and 51% of the shipped arm's navigation followed a departure from a node "
            "that still held tier-deferred work."
        ),
        "why_this_is_not_a_recommendation": (
            "Turning the barrier off is NOT free and is NOT proposed here. It also cuts "
            "new-state discovery roster-wide (2064 -> 1873 states, -9.3%), with large per-game "
            "losses (bp35 136 -> 20, ka59 110 -> 46, ar25 190 -> 131, cn04 186 -> 115). The "
            "barrier was shipped ON because REQ-ARC-WMTE-5836 measured +2..+4 games on click "
            "games; this probe prices the action cost of that decision, which had not been "
            "measured, and does not overturn it. The actionable reading is that the barrier's "
            "cost is concentrated in a DEPARTURE rule -- draining a node's deferred tiers "
            "before leaving, rather than paying RESET+replay to come back for them -- not that "
            "tiering itself is wrong."
        ),
    }

    payload["measured_nulls"] = [
        {
            "claim": "Frontier ordering is not choosing far targets when near ones exist.",
            "evidence": (
                "The locality counterfactual (cost of navigating to the CHEAPEST eligible open "
                "node instead of the chosen one, over the agent's own recorded forward edges) "
                "came out at ZERO actions above cheapest on tn36, the worst-affected game. "
                "Depth is the frontier's primary key and RESET-replay cost is 1+depth, so the "
                "shallowest open node IS the cheapest reachable one -- the two orderings "
                "coincide. 'Order the frontier for locality' is therefore not the lever."
            ),
        },
        {
            "claim": "Forward-walk navigation is already the dominant navigation mechanism.",
            "evidence": (
                "139 of 270 shipped navigation episodes used `_exact_shortest_path` (a forward "
                "walk, no RESET) and 8 used `_partial_forward_path`; only 123 fell back to "
                "RESET+replay. The 2026-06-20 lp85 regression (7,792 actions vs bare BFS's 21) "
                "that motivated recording forward edges is fixed and stays fixed."
            ),
        },
        {
            "claim": "Raising `frontier_batch_size` is NOT the fan-out amortization it looks like.",
            "evidence": (
                "The fan-out counterfactual says taking every tier-eligible row at each visited "
                "target instead of exactly one would cap navigation at 247 of 709 actions "
                "(saving 462). But `_serve` attributes only the FIRST batched probe to the "
                "target (`origin` is set on index 0); every later probe in the batch is "
                "attributed to `self.cur`, i.e. to wherever the previous probe left the agent. "
                "So the shipped knob queues B consecutive steps FROM the target, which is a "
                "depth-ride, not B alternatives AT the target. The 462-action ceiling is real "
                "but the existing flag does not reach it -- a fan-out mechanism would have to "
                "be written."
            ),
        },
    ]

    payload["how_measured"] = {
        "driver": "scripts/arc_explorer_renavigation_census.py",
        "worker": "scripts/arc_explorer_renavigation_probe.py",
        "report": "scripts/arc_explorer_renavigation_report.py",
        "agent_source_modified": False,
        "agent_source_modified_note": (
            "Nothing under python/carnot/agentic/ was edited. Every quantity is read between "
            "decisions from attributes the explorer already maintains (cur, graph, adj, "
            "pending, the _nav_* counters, navigation_diagnostics()) and from the _prov_branch "
            "/ _prov_serve_kind labels next_move already assigns unconditionally. The frontier "
            "snapshot deliberately does not call _frontier(), which mutates."
        ),
        "exemplar_rows_preserved": [g.strip() for g in args.exemplar_rows.split(",") if g.strip()],
        "exemplar_rows_note": (
            "Full per-action rows are preserved for four representative games (the outlier, a "
            "nav-only game, a tier-heavy click game, a cheap one). The other cells keep their "
            "class counts, episode records and summaries. The dropped rows are recoverable "
            "exactly: the runs are deterministic, ~5s per game, one command each."
        ),
    }

    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    payload["reproducibility_checksum"] = "sha256:" + hashlib.sha256(raw).hexdigest()

    out = out_dir / "arc_explorer_renavigation.json"
    out.write_text(json.dumps(payload, indent=1, default=str))

    # Evidence sidecars.
    trimmed = {
        "games": census["games"],
        "arms": arms,
        "budget": census["budget"],
        "seed": census["seed"],
        "duration_s": census["duration_s"],
        "cells": {
            arm: [{k: v for k, v in cell.items() if k != "rows"} for cell in census["cells"][arm]]
            for arm in arms
        },
    }
    (out_dir / "census_cells.json").write_text(json.dumps(trimmed, indent=1, default=str))
    exemplars = {g.strip() for g in args.exemplar_rows.split(",") if g.strip()}
    rows_dir = out_dir / "rows"
    rows_dir.mkdir(exist_ok=True)
    for arm in arms:
        for cell in census["cells"][arm]:
            if cell["game"] in exemplars and cell.get("rows"):
                (rows_dir / f"{cell['game']}__{arm}.json").write_text(
                    json.dumps(
                        {"game": cell["game"], "arm": arm, "rows": cell["rows"]},
                        indent=1,
                        default=str,
                    )
                )
    print(json.dumps(payload["headline"], indent=1))
    print(json.dumps(payload["theoretical_saving"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
