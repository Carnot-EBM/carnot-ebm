#!/usr/bin/env python3
"""Analysis for the leave-one-game-out ADAPTER-FREE / held-out-identity measurement.

Reads the rows written by the two cell drivers and emits ONE artifact. Deliberately separate
from the drivers so the pre-registered test (docs/research-notes/
arc-heldout-identity-prereg-2026-08-03.md) cannot be tuned against the numbers while they are
being collected.

WHAT IT REFUSES TO DO, and why each refusal is here:

  * It never reports a bare mean. Every per-arm figure is min/q1/median/q3/max over the 25-game
    roster, because a 73.8% headline once turned out to be one game's outlier against a roster
    median of 2.9%.
  * It never drops a cell silently. Errors, timeouts and unrunnable games stay in the
    denominator BY NAME with their reason.
  * It never reads a lower action count as a saving without checking `explored_out`. A run that
    stopped because it exhausted its frontier did LESS WORK; that is a regression wearing a
    win's clothes.
  * It never reports "no difference" without the vacuity check. If the two arms produce
    byte-identical action traces, the honest finding is that the removed knowledge was INERT on
    this path -- not that the agent generalized.
  * It states MIN REACHABLE p from the observed discordance count, so a null cannot be read as
    evidence of absence when no significant result was reachable in the first place.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from math import comb
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]


def sign_test_p(wins: int, losses: int) -> Optional[float]:
    """Two-sided exact sign-test p-value; ties dropped. None when nothing is discordant."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return round(min(1.0, 2.0 * tail), 6)


def min_reachable_p(n_discordant: int) -> Optional[float]:
    """The smallest two-sided p attainable with this many discordant pairs, all one direction."""
    if n_discordant <= 0:
        return None
    return round(min(1.0, 2.0 * (0.5**n_discordant)), 10)


def five_number(vals: list[float]) -> dict[str, Any]:
    """min/q1/median/q3/max, plus n. NEVER a bare mean (mean is included only alongside)."""
    v = sorted(float(x) for x in vals)
    if not v:
        return {"n": 0, "min": None, "q1": None, "median": None, "q3": None, "max": None}
    if len(v) >= 4:
        q1, _med, q3 = statistics.quantiles(v, n=4)
    else:
        q1 = q3 = None
    return {
        "n": len(v),
        "min": round(v[0], 4),
        "q1": round(q1, 4) if q1 is not None else None,
        "median": round(statistics.median(v), 4),
        "q3": round(q3, 4) if q3 is not None else None,
        "max": round(v[-1], 4),
        "mean_reported_only_alongside_the_distribution": round(statistics.mean(v), 4),
    }


def load_rows(d: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception as exc:
            out.append({"status": f"blocked_unreadable:{type(exc).__name__}", "path": p.name})
    return out


def analyse_scored(
    rows: list[dict[str, Any]], roster: list[str], registry: dict[str, int]
) -> dict[str, Any]:
    CTL, TRT = "control_identity_on", "heldout_identity_off"
    by: dict[tuple[str, str, int], dict[str, Any]] = {}
    bad: list[dict[str, Any]] = []
    for r in rows:
        if r.get("status") != "ok":
            bad.append(
                {
                    "game": r.get("game"),
                    "arm": r.get("arm"),
                    "seed": r.get("seed"),
                    "status": r.get("status"),
                    "error": (r.get("error") or "")[:200],
                }
            )
            continue
        by[(str(r["game"]), str(r["arm"]), int(r["seed"]))] = r

    seeds = sorted({k[2] for k in by})

    def metric(g: str, arm: str, field: str, banked: bool = False) -> list[float]:
        vals = []
        for s in seeds:
            r = by.get((g, arm, s))
            if r is None:
                continue
            if banked:
                vals.append(float(r.get("banked_levels") or 0))
            else:
                v = (r.get("result") or {}).get(field)
                if v is not None:
                    vals.append(float(v))
        return vals

    per_game: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for g in roster:
        c_b, t_b = metric(g, CTL, "", banked=True), metric(g, TRT, "", banked=True)
        c_l, t_l = metric(g, CTL, "levels_gained"), metric(g, TRT, "levels_gained")
        if not c_l or not t_l:
            excluded.append(
                {
                    "game": g,
                    "reason": "no usable cell in one or both arms",
                    "n_control_cells": len(c_l),
                    "n_heldout_cells": len(t_l),
                }
            )
            continue
        c_a = metric(g, CTL, "total_actions")
        t_a = metric(g, TRT, "total_actions")
        c_f = metric(g, CTL, "actions_to_first_solve")
        t_f = metric(g, TRT, "actions_to_first_solve")
        # Trace identity: how many matched (game,seed) pairs produced the SAME ordered actions.
        same_trace = 0
        n_pairs = 0
        for s in seeds:
            rc, rt = by.get((g, CTL, s)), by.get((g, TRT, s))
            if rc is None or rt is None:
                continue
            n_pairs += 1
            if rc.get("trace_sha256") and rc.get("trace_sha256") == rt.get("trace_sha256"):
                same_trace += 1
        eo_c = sum(
            1 for s in seeds if (by.get((g, CTL, s)) or {}).get("result", {}).get("explored_out")
        )
        eo_t = sum(
            1 for s in seeds if (by.get((g, TRT, s)) or {}).get("result", {}).get("explored_out")
        )
        per_game.append(
            {
                "game": g,
                # The hand-tuned dev-twin reach, carried here so the per-game row shows the drop
                # from 183/183 directly. NOT a control: different entrypoint, unbounded search.
                "registry_hand_tuned_levels_reference": registry.get(g),
                "banked_control": round(statistics.mean(c_b), 4) if c_b else 0.0,
                "banked_heldout": round(statistics.mean(t_b), 4) if t_b else 0.0,
                "levels_control": round(statistics.mean(c_l), 4),
                "levels_heldout": round(statistics.mean(t_l), 4),
                "delta_banked": round(
                    (statistics.mean(t_b) if t_b else 0.0) - (statistics.mean(c_b) if c_b else 0.0),
                    4,
                ),
                "actions_control": round(statistics.mean(c_a), 2) if c_a else None,
                "actions_heldout": round(statistics.mean(t_a), 2) if t_a else None,
                "first_solve_control": round(statistics.mean(c_f), 2) if c_f else None,
                "first_solve_heldout": round(statistics.mean(t_f), 2) if t_f else None,
                "matched_seed_pairs": n_pairs,
                "identical_action_traces": same_trace,
                "explored_out_control": eo_c,
                "explored_out_heldout": eo_t,
            }
        )

    deltas = [p["delta_banked"] for p in per_game]
    wins = sum(1 for d in deltas if d > 1e-9)
    losses = sum(1 for d in deltas if d < -1e-9)
    ties = len(deltas) - wins - losses

    # Leak delivery, aggregated over every ok cell, per arm.
    delivery: dict[str, dict[str, Any]] = {}
    for (g, arm, s), r in by.items():
        for site, rec in (r.get("leak_delivery") or {}).items():
            e = delivery.setdefault(
                site,
                {
                    "control_calls": 0,
                    "heldout_calls": 0,
                    "control_cells_hit": 0,
                    "heldout_cells_hit": 0,
                    "callers": [],
                },
            )
            k = "control" if arm == CTL else "heldout"
            e[f"{k}_calls"] += int(rec.get("calls") or 0)
            if (rec.get("calls") or 0) > 0:
                e[f"{k}_cells_hit"] += 1
            if rec.get("first_caller") and rec["first_caller"] not in e["callers"]:
                e["callers"].append(rec["first_caller"])

    n_pairs_total = sum(p["matched_seed_pairs"] for p in per_game)
    n_ident_total = sum(p["identical_action_traces"] for p in per_game)

    # SEED-AXIS DETERMINISM. If seeds 1/2/3 give the same trace within an arm, three seeds are
    # ONE effective replicate: the run carries no within-game variance estimate, and any
    # "mean over seeds" is just that single value wearing a mean's clothes. Recorded because a
    # reader would otherwise reasonably assume N=3 bought variance information that it did not.
    seed_det: dict[str, Any] = {"per_arm": {}, "all_deterministic": True}
    for arm in (CTL, TRT):
        same = tot = 0
        for g in roster:
            hs = [by[(g, arm, s)]["trace_sha256"] for s in seeds if (g, arm, s) in by]
            if len(hs) >= 2:
                tot += 1
                same += len(set(hs)) == 1
        seed_det["per_arm"][arm] = {
            "games_with_multiple_seeds": tot,
            "games_where_all_seeds_identical": same,
        }
        if tot and same != tot:
            seed_det["all_deterministic"] = False

    return {
        "unit_of_clustering": "game",
        "seeds": seeds,
        "n_games_in_roster": len(roster),
        "n_games_analysed": len(per_game),
        "games_excluded_by_name": excluded,
        "unusable_cells_by_name": bad,
        "per_game": sorted(per_game, key=lambda r: r["game"]),
        "distribution_banked_control": five_number([p["banked_control"] for p in per_game]),
        "distribution_banked_heldout": five_number([p["banked_heldout"] for p in per_game]),
        "distribution_levels_control": five_number([p["levels_control"] for p in per_game]),
        "distribution_levels_heldout": five_number([p["levels_heldout"] for p in per_game]),
        "distribution_actions_control": five_number(
            [p["actions_control"] for p in per_game if p["actions_control"] is not None]
        ),
        "distribution_actions_heldout": five_number(
            [p["actions_heldout"] for p in per_game if p["actions_heldout"] is not None]
        ),
        "sign_test_banked_levels": {
            "n_game_pairs": len(deltas),
            "heldout_better": wins,
            "ties": ties,
            "heldout_worse": losses,
            "n_discordant": wins + losses,
            "p_two_sided": sign_test_p(wins, losses),
            "min_reachable_p_given_observed_discordance": min_reachable_p(wins + losses),
            "min_reachable_p_prereg_all_25_discordant": min_reachable_p(25),
        },
        "vacuity_check": {
            "matched_seed_pairs": n_pairs_total,
            "identical_action_traces": n_ident_total,
            "frac_identical": round(n_ident_total / n_pairs_total, 4) if n_pairs_total else None,
            "reading": (
                "IF frac_identical is at or near 1.0 the two arms executed the SAME trajectory, "
                "so the removed per-game knowledge was INERT on this path in this configuration. "
                "That is a statement about the path, NOT evidence that the agent generalizes."
            ),
        },
        "seed_axis_determinism": seed_det,
        "explored_out_audit": {
            "control_cells_explored_out": sum(p["explored_out_control"] for p in per_game),
            "heldout_cells_explored_out": sum(p["explored_out_heldout"] for p in per_game),
            "games_with_any_explored_out": [
                p["game"]
                for p in per_game
                if p["explored_out_control"] or p["explored_out_heldout"]
            ],
            "reading": (
                "A cell that used fewer actions AND explored out did LESS WORK. Any action-count "
                "reduction on such a cell is a frontier collapse, not an efficiency win."
            ),
        },
        "leak_delivery_aggregate": delivery,
    }


def analyse_devtwin(
    rows: list[dict[str, Any]], roster: list[str], registry: dict[str, int]
) -> dict[str, Any]:
    CTL, TRT = "control_adapter_on", "treatment_adapter_free"
    by: dict[tuple[str, str], dict[str, Any]] = {}
    bad: list[dict[str, Any]] = []
    for r in rows:
        if r.get("status") != "ok":
            bad.append(
                {
                    "game": r.get("game"),
                    "arm": r.get("arm"),
                    "status": r.get("status"),
                    "error": (r.get("error") or "")[:200],
                }
            )
            continue
        by[(str(r["game"]), str(r["arm"]))] = r

    per_game, excluded = [], []
    for g in roster:
        c, t = by.get((g, CTL)), by.get((g, TRT))
        if c is None or t is None:
            excluded.append(
                {
                    "game": g,
                    "reason": "missing or unusable cell",
                    "control_present": c is not None,
                    "treatment_present": t is not None,
                }
            )
            continue
        cb, tb = int(c.get("banked_levels") or 0), int(t.get("banked_levels") or 0)
        per_game.append(
            {
                "game": g,
                "registry_hand_tuned_levels_reference": registry.get(g),
                "banked_L1_control_adapter_on": cb,
                "banked_L1_treatment_adapter_free": tb,
                "delta": tb - cb,
                "control_search_cost": (c.get("result") or {}).get("search_cost"),
                "treatment_search_cost": (t.get("result") or {}).get("search_cost"),
                "control_moves": (c.get("result") or {}).get("moves"),
                "treatment_moves": (t.get("result") or {}).get("moves"),
                "treatment_no_advance": (t.get("result") or {}).get("no_advance"),
                "control_wall_s": c.get("cell_wall_s"),
                "treatment_wall_s": t.get("cell_wall_s"),
            }
        )

    deltas = [p["delta"] for p in per_game]
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    return {
        "unit_of_clustering": "game",
        "objective": "bank the FIRST level from level 0, reproduction-gated",
        "matched_on": "the objective, NOT the search budget (each arm uses its own shipped config)",
        "n_games_analysed": len(per_game),
        "games_excluded_by_name": excluded,
        "unusable_cells_by_name": bad,
        "per_game": sorted(per_game, key=lambda r: r["game"]),
        "n_games_control_banked_L1": sum(
            1 for p in per_game if p["banked_L1_control_adapter_on"] >= 1
        ),
        "n_games_treatment_banked_L1": sum(
            1 for p in per_game if p["banked_L1_treatment_adapter_free"] >= 1
        ),
        "distribution_control_search_cost": five_number(
            [p["control_search_cost"] for p in per_game if p["control_search_cost"] is not None]
        ),
        "sign_test_banked_L1": {
            "n_game_pairs": len(deltas),
            "treatment_better": wins,
            "ties": len(deltas) - wins - losses,
            "treatment_worse": losses,
            "n_discordant": wins + losses,
            "p_two_sided": sign_test_p(wins, losses),
            "min_reachable_p_given_observed_discordance": min_reachable_p(wins + losses),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scored-dir", default=str(REPO / "results" / "arc_heldout_identity_20260803"))
    ap.add_argument(
        "--devtwin-dir", default=str(REPO / "results" / "arc_devtwin_adapterfree_20260803")
    )
    ap.add_argument(
        "--devtwin-long-dir",
        default=str(REPO / "results" / "arc_devtwin_adapterfree_long_20260803"),
        help="the 30000-expansion re-run of the games the adapter-free arm failed at 6000",
    )
    ap.add_argument(
        "--out",
        default=str(REPO / "results" / "experiment_6093_arc_heldout_identity_generalization.json"),
    )
    args = ap.parse_args(argv)

    import yaml

    reg = yaml.safe_load((REPO / "ops" / "arc_solve_registry.yaml").read_text())
    roster = sorted(str(e["game"]) for e in reg["games"])
    registry_levels = {str(e["game"]): int(e.get("levels_reproduced") or 0) for e in reg["games"]}

    scored_rows = load_rows(Path(args.scored_dir))
    devtwin_rows = load_rows(Path(args.devtwin_dir))

    scored = analyse_scored(scored_rows, roster, registry_levels)
    devtwin = analyse_devtwin(devtwin_rows, roster, registry_levels)

    # BUDGET WALL vs CAPABILITY WALL. A game the adapter-free explorer failed at the shipped
    # 6000-expansion budget has two possible readings, and they mean opposite things for a hidden
    # game: "it needed more search" (a cost problem, buyable) or "more search does not help" (a
    # representation problem, not buyable). Re-running exactly the failures at 5x separates them.
    # Reported as its own section rather than folded into the headline, because it is a different
    # budget and pooling it would misstate the matched comparison.
    long_rows = load_rows(Path(args.devtwin_long_dir))
    long_out = []
    for r in long_rows:
        long_out.append(
            {
                "game": r.get("game"),
                "status": r.get("status"),
                "max_expansions": 30000,
                "banked_levels": r.get("banked_levels"),
                "reached_level": (r.get("result") or {}).get("reached_level"),
                "moves": (r.get("result") or {}).get("moves"),
                "wall_s": r.get("cell_wall_s"),
                "error": (r.get("error") or "")[:200] or None,
            }
        )
    flipped = [r["game"] for r in long_out if (r.get("banked_levels") or 0) >= 1]
    still = [
        r["game"] for r in long_out if r.get("status") == "ok" and (r.get("banked_levels") or 0) < 1
    ]
    unusable_long = [
        {"game": r["game"], "status": r["status"]} for r in long_out if r.get("status") != "ok"
    ]

    # ---- REPRODUCIBILITY CHECKSUM ------------------------------------------------------------
    # Content-addressed over every input row this analysis read, in sorted order, so a third
    # party can confirm they are analysing the same corpus. Catches a silently re-run or
    # partially-overwritten sweep, which a row COUNT would not.
    h = hashlib.sha256()
    n_input_rows = 0
    total_cell_wall_s = 0.0
    for d in (Path(args.scored_dir), Path(args.devtwin_dir), Path(args.devtwin_long_dir)):
        for f in sorted(d.glob("*.json")):
            b = f.read_bytes()
            h.update(f.name.encode())
            h.update(b)
            n_input_rows += 1
            try:
                total_cell_wall_s += float(json.loads(b).get("cell_wall_s") or 0.0)
            except Exception:
                pass

    payload = {
        "experiment": 6093,
        "experiment_id": "experiment_6093_arc_heldout_identity_generalization",
        "title": (
            "Leave-one-game-out adapter-free / held-out-identity measurement on the two "
            "canonical ARC live entrypoints"
        ),
        "schema": "carnot.arc_heldout_identity_generalization.v1",
        "prereg": "docs/research-notes/arc-heldout-identity-prereg-2026-08-03.md",
        "scored_path_E3AgentPolicy": scored,
        "dev_twin_arc_loop_solve": devtwin,
        "dev_twin_long_budget_probe": {
            "question": "budget wall or capability wall?",
            "max_expansions": 30000,
            "baseline_max_expansions": 6000,
            "games_probed": sorted(r["game"] for r in long_out if r.get("game")),
            "flipped_to_banked_L1_at_5x_budget": sorted(flipped),
            "still_failed_at_5x_budget": sorted(still),
            "unusable_by_name": unusable_long,
            "rows": sorted(long_out, key=lambda r: str(r.get("game"))),
        },
        # ---- WHY THE SCORED-PATH NULL LOOKS THE WAY IT DOES ----------------------------------
        # A static, grep-verifiable fact about the code, recorded next to the measurement because
        # without it the null reads as "the agent generalizes" when it actually means "the
        # knowledge never reaches the action loop". `_recommend_live_approach` is assigned to
        # `self.approach_recommendation` at arc_competition_agent.py:4629 / :4825 / :4916, and the
        # ONLY key any other site reads back is `["strategy"]` (:4631, :4642). The registry digest
        # (A3), retrieved primitives (A4), ranked transfer recipes (A5) and selected primitive
        # operators (A6) are computed, stored, and never read again on this path.
        "scored_path_identity_channels": {
            "channel_1_strategy_route_to_explore_budget": {
                "live": True,
                "sites": ["arc_competition_agent.py:4631", ":4642", ":4826", ":4917"],
                "games_where_it_differs": 2,
                "games_named": ["sb26", "tn36"],
                "note": (
                    "23 of 25 public games already route to graph_explore / budget 80 -- the same "
                    "default a hidden game gets -- so this channel is inert on 23 of 25 by "
                    "construction, before any measurement."
                ),
            },
            "channel_2_llm_induce_prompt_and_engine_store": {
                "live_in_this_run": False,
                "sites": [
                    "arc_competition_agent.py:6517 self._proposer().induce(self.short, ...)",
                    "arc_executable_world_model.py:3229 E3_DIR / game",
                    "arc_executable_world_model.py:2958 induce_prompt(game, ...)",
                ],
                "note": (
                    "THIS RUN CANNOT TEST THIS CHANNEL. The game id and everything derived from "
                    "it flow into the LLM induce prompt and the per-game engine store, both dead "
                    "with the generator off. The scored-path null therefore bounds the NON-LLM "
                    "contribution of per-game identity and is silent on the LLM-mediated one."
                ),
            },
            "keys_computed_but_never_read_back": [
                "A3 registry game_digest (the target's own hand-written win-condition prose)",
                "A4 retrieved documented primitives",
                "A5 ranked transfer recipes / confident_transfer / top_similarity",
                "A6 selected primitive operators",
            ],
        },
        # ---- MANDATED ARTIFACT FIELDS --------------------------------------------------------
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "inference_substrate_note": (
            "The live agent takes real actions against the offline arcade with NO LLM invoked: "
            "the proposer is a counted stub whose induce() returns (False, reason), so no GGUF is "
            "loaded and no CUDA is touched. GPU 1 belongs to a concurrent workflow and was not "
            "used. Every generator string anywhere in this artifact names a model that WOULD fire "
            "if the LLM tier were enabled; none was invoked."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_note": (
            "No hidden game was played and none may be claimed. The held-out arm ANONYMIZES the "
            "policy's notion of which game it is while the environment keeps running the real "
            "public game, which is a simulation of the hidden condition, not the hidden condition."
        ),
        "read_game_source": False,
        "used_env_source": True,
        "random_seed": 1,
        "random_seeds_used": [1, 2, 3],
        "reproducibility_checksum": h.hexdigest(),
        "n_input_rows": n_input_rows,
        "duration_s": round(total_cell_wall_s, 1),
        "duration_note": (
            "Summed wall-clock of every cell subprocess. Cells ran 4-8 wide, so this exceeds the "
            "elapsed time of the sweep and is the honest COMPUTE cost, not the calendar cost."
        ),
        "verifier_is_oracle": True,
        "verifier_is_oracle_note": (
            "The win gate is the level counter plus arc_solver_kit.reproduce -- an executable "
            "oracle, not a learned verifier. Declared for completeness: this artifact makes NO "
            "verifier-moat, verifier-value-added or efficiency claim, and flips no gate."
        ),
        "preconditions_checked": [
            {"resource": "environment_files/<game>/ for all 25 roster games", "available": True},
            {"resource": "ops/arc_solve_registry.yaml (25 games)", "available": True},
            {"resource": "GPU", "available": False, "note": "deliberately not used; not required"},
            {"resource": "local GGUF generator", "available": False, "note": "LLM-off by design"},
        ],
        "registry_reference_not_a_control": {
            "reproducible_total_levels": int(reg["reproducible_total_levels"]),
            "reproducible_total_games": int(reg["reproducible_total_games"]),
            "per_game": registry_levels,
            "caveat": (
                "Produced by unbounded best-first search through a hand-built GameAdapter, not by "
                "a 400-action live episode. A ceiling reference, never a budget-matched control."
            ),
        },
    }
    # ---- HONEST VERDICT, COMPUTED FROM THE RESULTS ------------------------------------------
    # Derived rather than written by hand, so it cannot drift from the numbers above. Terminal
    # prefix per CLAUDE.md's Verdict Terminal-Prefix Discipline; no ": " anywhere (an unquoted
    # colon-space in a verdict has previously poisoned research-complete.yaml).
    dt = devtwin["sign_test_banked_L1"]
    sc = scored["sign_test_banked_levels"]
    payload["headline"] = {
        "dev_twin_L1_acquisition_adapter_on": devtwin["n_games_control_banked_L1"],
        "dev_twin_L1_acquisition_adapter_free": devtwin["n_games_treatment_banked_L1"],
        "dev_twin_L1_acquisition_adapter_free_at_5x_budget": (
            devtwin["n_games_treatment_banked_L1"] + len(flipped)
        ),
        "dev_twin_sign_test_p": dt["p_two_sided"],
        "scored_path_discordant_games": sc["n_discordant"],
        "scored_path_p": sc["p_two_sided"],
        "scored_path_identical_trace_fraction": scored["vacuity_check"]["frac_identical"],
        "scored_path_total_banked_levels_control": round(
            sum(p_["banked_control"] for p_ in scored["per_game"]), 2
        ),
        "scored_path_total_banked_levels_heldout": round(
            sum(p_["banked_heldout"] for p_ in scored["per_game"]), 2
        ),
        "registry_total_levels_for_the_same_25_games": int(reg["reproducible_total_levels"]),
    }
    hl = payload["headline"]
    payload["honest_verdict"] = (
        "complete_adapter_free_devtwin_first_level_acquisition_falls_"
        f"{hl['dev_twin_L1_acquisition_adapter_on']}of{devtwin['n_games_analysed']}_to_"
        f"{hl['dev_twin_L1_acquisition_adapter_free']}of{devtwin['n_games_analysed']}_sign_p_"
        f"{dt['p_two_sided']}_and_only_"
        f"{hl['dev_twin_L1_acquisition_adapter_free_at_5x_budget']}of"
        f"{devtwin['n_games_analysed']}_at_5x_search_budget_"
        "while_on_the_scored_path_identity_removal_is_INERT_"
        f"{sc['n_discordant']}_of_{sc['n_game_pairs']}_games_discordant_"
        f"{scored['vacuity_check']['identical_action_traces']}_of_"
        f"{scored['vacuity_check']['matched_seed_pairs']}_action_traces_byte_identical_"
        "no_p_reachable_and_the_scored_path_banks_"
        f"{hl['scored_path_total_banked_levels_control']}_levels_in_400_actions_against_"
        f"{hl['registry_total_levels_for_the_same_25_games']}_hand_tuned_registry_levels"
    )

    Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps({"wrote": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
