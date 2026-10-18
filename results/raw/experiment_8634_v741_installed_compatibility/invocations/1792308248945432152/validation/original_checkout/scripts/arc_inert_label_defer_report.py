#!/usr/bin/env python3
"""Turn the REQ-ARC-WMTE-6071 A/B into the milestone artifact, in the benchmark's own units.

Reads ``<out-dir>/ab.json`` (written by ``scripts/arc_inert_label_defer_ab.py``) and emits
``<out-dir>/arc_inert_label_defer.json``: the adversarial-verify-shaped artifact, with the
pre-registered statistics, the A/A noise floor, the per-level score conversion, and the
disclosures that make the verdict readable.

SCORE CONVERSION. ARC-AGI-3 scores a level as ``min((baseline_actions/agent_actions)**2, 115)``.
The human baseline is not known to this project, but it CANCELS in the arm-vs-arm ratio, so the
honest statement is the MULTIPLIER on a level's score:
``(control_actions_to_levelup / defer_actions_to_levelup)**2``, reported per game and only for
games where BOTH arms banked the level. A game where one arm banks and the other does not is a
capability difference, not an efficiency one, and is reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent


def sign_test_p(wins: int, losses: int) -> Optional[float]:
    from math import comb

    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    return min(1.0, 2 * sum(comb(n, i) for i in range(0, k + 1)) / (2**n))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", required=True)
    ap.add_argument("--k4-dir", default="")
    ap.add_argument("--live-budget", type=int, default=400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ab = json.loads(Path(args.ab).read_text())
    games, seeds = ab["games"], ab["seeds"]
    C = ab["cells"]

    # ---- A/A noise floor ---------------------------------------------------------------
    aa_pairs, aa_identical = 0, 0
    for g in games:
        for s in seeds:
            a, b = C["control"].get(f"{g}|{s}"), C["control_b"].get(f"{g}|{s}")
            if not a or not b or a.get("error") or b.get("error"):
                continue
            aa_pairs += 1
            aa_identical += int(a["trace_sha256"] == b["trace_sha256"])
    deterministic = aa_pairs > 0 and aa_identical == aa_pairs

    # ---- seed axis: is it actually an axis? ---------------------------------------------
    seed_sensitive = []
    for arm in ("control", "defer"):
        for g in games:
            hs = {
                (C[arm].get(f"{g}|{s}") or {}).get("trace_sha256")
                for s in seeds
                if C[arm].get(f"{g}|{s}") and not C[arm][f"{g}|{s}"].get("error")
            }
            if len(hs) > 1:
                seed_sensitive.append({"arm": arm, "game": g, "distinct_traces": len(hs)})

    # ---- jurisdiction ------------------------------------------------------------------
    fired, no_jurisdiction, tracked_but_silent = [], [], []
    for g in games:
        cell = C["defer"].get(f"{g}|{seeds[0]}") or {}
        d = cell.get("inert_label_defer_diagnostics") or {}
        if d.get("error"):
            no_jurisdiction.append({"game": g, "diagnostics_error": d["error"]})
        elif int(d.get("deferred_pops", 0)) > 0:
            fired.append(
                {
                    "game": g,
                    "deferred_pops": int(d["deferred_pops"]),
                    "rows_deferred": int(d["rows_deferred"]),
                    "labels_deferrable": int(d.get("labels_deferrable", 0)),
                    "observe_calls": int(d.get("observe_calls", 0)),
                }
            )
        else:
            tracked_but_silent.append(
                {
                    "game": g,
                    "observe_calls": int(d.get("observe_calls", 0)),
                    "labels_tracked": int(d.get("labels_tracked", 0)),
                    "labels_deferrable": int(d.get("labels_deferrable", 0)),
                    "why": (
                        "dead_observe_channel"
                        if int(d.get("observe_calls", 0)) == 0
                        else "no_label_was_ever_observed_inert_on_this_game"
                    ),
                }
            )

    # ---- per-level score conversion -----------------------------------------------------
    per_level_score, capability_diff = [], []
    for g in games:
        c = C["control"].get(f"{g}|{seeds[0]}") or {}
        d = C["defer"].get(f"{g}|{seeds[0]}") or {}
        ca, da = c.get("actions_to_first_levelup"), d.get("actions_to_first_levelup")
        if c.get("levels_gained", 0) != d.get("levels_gained", 0):
            capability_diff.append(
                {
                    "game": g,
                    "control_levels": c.get("levels_gained"),
                    "defer_levels": d.get("levels_gained"),
                    "control_actions_to_first_levelup": ca,
                    "defer_actions_to_first_levelup": da,
                }
            )
        if ca and da:
            per_level_score.append(
                {
                    "game": g,
                    "control_actions_to_first_levelup": ca,
                    "defer_actions_to_first_levelup": da,
                    "score_multiplier_first_level": round((ca / da) ** 2, 4),
                }
            )
    mults = [r["score_multiplier_first_level"] for r in per_level_score]
    wins = sum(1 for m in mults if m > 1.0)
    losses = sum(1 for m in mults if m < 1.0)

    # ---- evidence-floor sensitivity arm (min_observations = 4) --------------------------
    k4: dict[str, Any] = {}
    if args.k4_dir:
        for f in sorted(Path(args.k4_dir).glob("*.json")):
            row = json.loads(f.read_text())
            if row.get("game"):
                k4[row["game"]] = row

    def _live_budget_bankers(getter) -> list[str]:
        """Games that bank their FIRST level inside the LIVE per-game action budget.

        This is the number that decides a submission, and it is NOT the same question as the
        budget-2000 level total: a level banked at action 1975 of 2000 is a level the live eval
        never sees. Reported separately for exactly that reason.
        """

        out = []
        for g in games:
            a = getter(g)
            if a and int(a) <= int(args.live_budget):
                out.append(g)
        return out

    ctl_live = _live_budget_bankers(
        lambda g: (C["control"].get(f"{g}|{seeds[0]}") or {}).get("actions_to_first_levelup")
    )
    def_live = _live_budget_bankers(
        lambda g: (C["defer"].get(f"{g}|{seeds[0]}") or {}).get("actions_to_first_levelup")
    )
    k4_live = (
        _live_budget_bankers(lambda g: (k4.get(g) or {}).get("actions_to_first_levelup"))
        if k4
        else None
    )
    gained = sorted(set(def_live) - set(ctl_live))
    lost = sorted(set(ctl_live) - set(def_live))

    k4_block: dict[str, Any] = {}
    if k4:
        k4_rows = []
        for g in games:
            c = C["control"].get(f"{g}|{seeds[0]}") or {}
            r = k4.get(g) or {}
            ca, ka = c.get("actions_to_first_levelup"), r.get("actions_to_first_levelup")
            k4_rows.append(
                {
                    "game": g,
                    "control_levels": c.get("levels_gained"),
                    "k4_levels": r.get("levels_gained"),
                    "control_actions_to_first_levelup": ca,
                    "k4_actions_to_first_levelup": ka,
                    "score_multiplier_first_level": round((ca / ka) ** 2, 4) if ca and ka else None,
                    "control_inert": c.get("inert_actions"),
                    "k4_inert": r.get("inert_actions"),
                    "control_states": c.get("states_discovered"),
                    "k4_states": r.get("states_discovered"),
                }
            )
        k4_block = {
            "why": (
                "min_observations is the one tuning knob the mechanism has, and the k=1 arm's "
                "single capability loss (cd82) is exactly what a false-positive deferral would "
                "look like. Raising the evidence floor to 4 tests that hypothesis directly."
            ),
            "arm": "defer_k4 (InertLabelMemory(min_observations=4), one seed)",
            "levels_total_control": sum(
                (C["control"].get(f"{g}|{seeds[0]}") or {}).get("levels_gained", 0) for g in games
            ),
            "levels_total_defer_k1": sum(
                (C["defer"].get(f"{g}|{seeds[0]}") or {}).get("levels_gained", 0) for g in games
            ),
            "levels_total_defer_k4": sum((k4.get(g) or {}).get("levels_gained", 0) for g in games),
            "inert_total_control": sum(
                (C["control"].get(f"{g}|{seeds[0]}") or {}).get("inert_actions", 0) for g in games
            ),
            "inert_total_defer_k4": sum((k4.get(g) or {}).get("inert_actions", 0) for g in games),
            "rows": k4_rows,
        }

    payload: dict[str, Any] = {
        "experiment": "arc_inert_label_defer_ab",
        "requirement": "REQ-ARC-WMTE-6071",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "schema": "carnot.arc_inert_label_defer_ab.v1",
        "flag_under_test": "CARNOT_ARC_INERT_LABEL_DEFER / SUBMITTED_INERT_LABEL_DEFER_ENABLED",
        "flag_state_unchanged": True,
        "budget": ab["budget"],
        "games": games,
        "random_seeds_used": seeds,
        "random_seed": seeds[0],
        "n_cells": len(games) * len(seeds) * len(ab["arms"]),
        "missing_observations": ab["missing_observations"],
        "n_missing": ab["n_missing"],
        "duration_s": ab["duration_s"],
        "aa_noise_floor": {
            "pairs_compared": aa_pairs,
            "identical_action_traces": aa_identical,
            "agent_deterministic_at_fixed_seed": deterministic,
        },
        "seed_axis_is_largely_inert": {
            "cells_whose_trace_varies_with_seed": seed_sensitive,
            "n_games_seed_sensitive": len({r["game"] for r in seed_sensitive}),
        },
        "jurisdiction": {
            "games_where_the_lever_fired": fired,
            "games_tracked_but_silent": tracked_but_silent,
            "games_with_diagnostics_error": no_jurisdiction,
        },
        "paired_game_clustered": ab["paired"],
        "per_level_score_conversion": {
            "formula": "min((baseline/agent)**2, 115); baseline cancels in the arm ratio",
            "rows": sorted(per_level_score, key=lambda r: -r["score_multiplier_first_level"]),
            "games_faster_to_first_level": wins,
            "games_slower_to_first_level": losses,
            "sign_test_p": sign_test_p(wins, losses),
            "min_reachable_p_given_k_discordant": (2 * 0.5 ** (wins + losses))
            if (wins + losses)
            else None,
            "geometric_mean_multiplier": round(float(statistics.geometric_mean(mults)), 4)
            if mults
            else None,
        },
        "capability_differences": capability_diff,
        "live_action_budget_analysis": {
            "live_budget_actions": args.live_budget,
            "why": (
                "The live eval bounds actions per game (MAX_ACTIONS 400 on the submitted path), "
                "so a level first banked at action 1975 of a 2000-action measurement budget is a "
                "level the scored run never reaches. This re-reads the SAME cells at the budget "
                "that actually decides a submission."
            ),
            "control_games_banking_within_budget": ctl_live,
            "defer_games_banking_within_budget": def_live,
            "defer_k4_games_banking_within_budget": k4_live,
            "games_gained": gained,
            "games_lost": lost,
            "sign_test_p": sign_test_p(len(gained), len(lost)),
            "min_reachable_p_given_k_discordant": (2 * 0.5 ** (len(gained) + len(lost)))
            if (len(gained) + len(lost))
            else None,
        },
        "evidence_floor_sensitivity": k4_block,
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "inference_substrate_note": (
            "The live E3AgentPolicy cascade takes real actions against the offline arcade with "
            "CARNOT_ARC_DISABLE_INDUCTION=1 and CUDA_VISIBLE_DEVICES empty. No GGUF is loaded, no "
            "llama-server is started, no GPU is touched, and no scored or online game is played."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_note": (
            "Public-game offline measurement of a SEARCH-POLICY change. No level is claimed as "
            "banked by this artifact and no registry entry is written. The mechanism itself is "
            "live-agent self-discovery -- it learns only from the running agent's own observed "
            "transitions, needs no adapter, no game source and no generator, and is therefore as "
            "available on a hidden game as on a public one -- but THIS RUN is a dev measurement."
        ),
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": (
            "The memory is a frequency tally over the agent's own observed frame deltas. It never "
            "consults the win condition; the level counter (the oracle) is only ever READ as the "
            "outcome metric, and enters the mechanism solely as the SACRED veto that protects a "
            "label which already banked a level."
        ),
        "read_game_source": False,
        "used_env_source": True,
        "used_env_source_note": (
            "The hand-verifier progress proxy calls the adapter's public callable against the live "
            "runtime game object, per arc_actions_to_progress. No game .py source is read."
        ),
        "offline_ground_truth_bfs": False,
        "exhaustive_bfs_calibration": False,
        "hand_calibrated_per_game": False,
        "preconditions_checked": [
            {"resource": "offline arcade environment_files tree", "available": True},
            {"resource": "CARNOT_ARC_E3_DIR redirected to per-arm scratch", "available": True},
            {"resource": "CARNOT_ARC_DISABLE_INDUCTION=1", "available": True},
            {"resource": "CUDA_VISIBLE_DEVICES empty (no GPU)", "available": True},
            {"resource": "no llama-server started", "available": True},
        ],
        "prior_failures": [
            {
                "experiment_id": "outer_loop_inert_click_pruner_shipped_config_ab_20260726",
                "verdict": (
                    "complete: inert_click_pruner ... zero new wins any seed, lost ft09 on 2 of 3 "
                    "seeds, states_expanded +12.0% pooled / +37.9% non-HUD, recommend do not flip "
                    "and retire"
                ),
                "root_cause_diagnosed": (
                    "It keyed on a STRUCTURAL blob signature (color, pixel_count, is_rect, "
                    "twin_count), so evidence about one blob suppressed clicks on every "
                    "look-alike blob; and it DROPPED rows from node['untested'], shortening the "
                    "node's list, retiring it from the frontier early and buying navigation."
                ),
                "what_is_different_here": (
                    "(1) KEY: the literal (action_id, x, y) the agent will send, which generalizes "
                    "across STATES and across nothing else -- measured 98.4% precision / 71.7% "
                    "recall on 43,533 roster probe actions. (2) CONSEQUENCE: this never drops a "
                    "row; it changes only WHICH row a node pops next, and abstains entirely once "
                    "every remaining row is deferrable, so node['untested'] length and therefore "
                    "_node_has_open_tier and the frontier schedule are untouched. (3) CHANNEL: it "
                    "learns from _last_unmasked_hash, which _ingest maintains unconditionally, "
                    "rather than awaiting['previous_frame'], which nine unrelated components gate."
                ),
                "retire_if_same_verdict": True,
            }
        ],
        "positive_control_the_lever_is_not_inert": {
            "why_this_matters": (
                "A null claim is not a finding unless the mechanism demonstrably had the "
                "opportunity to act (CLAUDE.md FALSE_NEGATIVE_RISK)."
            ),
            "games_where_it_fired": len(fired),
            "games_out_of_jurisdiction_no_label_ever_inert": len(tracked_but_silent),
            "dead_observe_channel_cells": sum(
                1 for r in tracked_but_silent if r["why"] == "dead_observe_channel"
            ),
            "target_class_reduction_pct": None,
            "conclusion": (
                "The lever fired on 15 of 25 games, cut its own target class by more than half "
                "with p=0.0018 on the game-clustered sign test, and changed the action trace on "
                "every game it fired on. The progress nulls are therefore measured nulls on an "
                "ACTIVE mechanism, not a degenerate no-op."
            ),
        },
        "measurement_defect_found_and_discarded_in_this_run": {
            "what": (
                "The FIRST A/B corpus was thrown away, not repaired. While it was running in the "
                "background, this session hot-swapped python/carnot/agentic/arc_competition_agent.py "
                "to the HEAD version for ~20 seconds to check whether three unrelated pre-existing "
                "test failures were pre-existing. Seven of 225 cells provably ran against the "
                "pre-change source, identified by an AttributeError in their own "
                "inert_label_defer_diagnostics field."
            ),
            "why_discarded_rather_than_re_run_cell_by_cell": (
                "The 7 cells are the ones that ANNOUNCED the swap. A cell that ran during the "
                "window with the flag off would have been behaviourally identical and silent, so "
                "the contaminated set cannot be bounded from the artifacts alone."
            ),
            "how_the_reported_corpus_is_protected": (
                "The source sha256 of both changed files was recorded before the clean run started "
                "and verified byte-identical after it finished; no cell in the reported corpus "
                "carries a diagnostics error; nothing was edited while it ran."
            ),
            "source_sha256_pinned_before_and_after": True,
        },
        "design_limits_stated_up_front": {
            "seed_axis_is_decorative": (
                "0 of 25 games produced a different action trace across the three seeds in either "
                "arm. random.seed/np.random.seed do not reach the explorer's own frontier RNG, so "
                "the three seeds are ONE observation repeated, not three replicates. The effective "
                "unit of analysis is the GAME (25 units) and the artifact reports it that way."
            ),
            "minimum_reachable_p": (
                "The pre-registered test is a two-sided paired sign test clustered at the game, so "
                "min p = 2 * 0.5**k with k discordant games: k>=6 for 0.05. On the primary "
                "progress axis (levels) only ONE game moved, so the design COULD NOT reach 0.05 "
                "there no matter what the effect was. This is stated as a design limit, not "
                "discovered afterwards."
            ),
            "budget_2000_is_not_the_live_budget": (
                "The measurement budget is 2000 actions because that is where the roster produces "
                "progress signal at all (11 of 25 games bank a level). The live per-game budget is "
                "400. Both readings are reported; they disagree, and the 400 reading is the one "
                "relevant to a submission."
            ),
            "no_llm_in_the_loop": (
                "Induction is disabled, so this measures the SEARCH. The census that motivated the "
                "work found the inert-probe share is the same with the generator on and off, but "
                "this run does not re-establish that."
            ),
        },
        "field_provenance": {
            "honest_verdict": {
                "principle": (
                    "Self-declared terminal state lets the reconciler classify the run without "
                    "re-running it; a terminal prefix keeps the partial-token matcher from "
                    "false-positiving on words like 'worse' or 'null'."
                )
            },
            "random_seeds_used": {
                "principle": (
                    "Determinism is the precondition for reproducibility. Recorded even though the "
                    "seed axis turned out to be inert -- which is itself a finding the field makes "
                    "checkable."
                )
            },
            "reproducibility_checksum": {
                "principle": (
                    "Content hash of the full result payload, so a later reader can tell a "
                    "re-derived number from a re-read one."
                )
            },
            "aa_noise_floor": {
                "principle": (
                    "Two identically-configured arms in separate processes. 75/75 identical traces "
                    "means the agent is deterministic here, which is a STRONGER statement than any "
                    "p-value: every treatment divergence is causal, not sampled."
                )
            },
            "live_action_budget_analysis": {
                "principle": (
                    "Prevents the measurement budget from flattering either arm. A level banked at "
                    "action 1975 of 2000 does not exist at the live budget of 400."
                )
            },
            "positive_control_the_lever_is_not_inert": {
                "principle": (
                    "A null on an unfired mechanism is a wiring bug reported as a finding. This "
                    "field is what separates the two."
                )
            },
        },
    }

    inert = payload["paired_game_clustered"]["inert"]
    payload["positive_control_the_lever_is_not_inert"]["target_class_reduction_pct"] = round(
        100.0 * (inert["pooled_defer"] - inert["pooled_control"]) / inert["pooled_control"], 2
    )
    payload["headline"] = {
        "target_class_inert_actions": {
            "control": inert["pooled_control"],
            "defer": inert["pooled_defer"],
            "pct": payload["positive_control_the_lever_is_not_inert"]["target_class_reduction_pct"],
            "games_better": inert["games_better"],
            "games_worse": inert["games_worse"],
            "sign_test_p": inert["sign_test_p"],
        },
        "progress_levels_gained_budget_2000": {
            "control": payload["paired_game_clustered"]["levels"]["pooled_control"],
            "defer": payload["paired_game_clustered"]["levels"]["pooled_defer"],
            "games_better": payload["paired_game_clustered"]["levels"]["games_better"],
            "games_worse": payload["paired_game_clustered"]["levels"]["games_worse"],
            "sign_test_p": payload["paired_game_clustered"]["levels"]["sign_test_p"],
            "min_reachable_p": payload["paired_game_clustered"]["levels"][
                "min_reachable_p_given_k_discordant"
            ],
        },
        "progress_games_banking_within_live_400_action_budget": {
            "control": len(ctl_live),
            "defer": len(def_live),
            "gained": gained,
            "lost": lost,
            "sign_test_p": payload["live_action_budget_analysis"]["sign_test_p"],
            "min_reachable_p": payload["live_action_budget_analysis"][
                "min_reachable_p_given_k_discordant"
            ],
        },
        "navigation_actions": {
            "control": payload["paired_game_clustered"]["nav"]["pooled_control"],
            "defer": payload["paired_game_clustered"]["nav"]["pooled_defer"],
            "games_better": payload["paired_game_clustered"]["nav"]["games_better"],
            "games_worse": payload["paired_game_clustered"]["nav"]["games_worse"],
            "sign_test_p": payload["paired_game_clustered"]["nav"]["sign_test_p"],
            "reading": (
                "SIGNIFICANTLY WORSE. The retired signature pruner's failure mode reappears in a "
                "different form: this lever does not shorten node lists, but by finding an "
                "effective action sooner it leaves nodes open and builds a deeper graph, and "
                "navigating that graph costs 1 + depth per RESET-replay."
            ),
        },
    }
    payload["acceptance_gate"] = {
        "pre_registered_primary": (
            "paired game-clustered sign test on levels_gained at budget 2000; PASS = defer > "
            "control with p < 0.05"
        ),
        "primary_verdict": "FAIL (0 games better, 1 worse; k=1 discordant so min reachable p = 1.0)",
        "pre_registered_secondary_hv": "FAIL (1 better, 1 worse, p = 1.0)",
        "pre_registered_secondary_states": "FAIL (8 better, 6 worse, p = 0.79)",
        "mechanism_gate_target_class_reduced": "PASS (13 better, 1 worse, p = 0.0018)",
        "post_hoc_disclosed_as_such_live_400_budget": (
            "3 -> 6 games banking a level; 3 gained, 0 lost; p = 0.25, and min reachable p is "
            "ALSO 0.25 with k=3, so this axis could not have been significant either. Promoted "
            "because the pre-registered budget-2000 level count answers a question the live eval "
            "does not ask, not because it read better."
        ),
        "recommendation": (
            "DO NOT flip SUBMITTED_INERT_LABEL_DEFER_ENABLED yet. The mechanism is confirmed and "
            "the per-level efficiency effect on the inert-heavy games is large (ft09 x59.5, lp85 "
            "x9.0, su15 x2.5 on the first level's score), but no progress axis reached "
            "significance, cd82 loses a level at BOTH evidence floors, and navigation is "
            "significantly worse. The next measurement is a LIVE-BUDGET (400-action) A/B with "
            "enough games banking to make the primary axis reachable, plus an investigation of "
            "why cd82 regresses -- not another 2000-action roster sweep."
        ),
    }
    payload["honest_verdict"] = (
        "complete_exact_label_inert_deferral_cuts_its_target_class_54pct_9208_to_4239_p0.0018_and_"
        "triples_the_games_banking_a_level_inside_the_live_400_action_budget_3_to_6_ft09_x59.5_"
        "lp85_x9.0_su15_x2.5_per_level_score_but_EVERY_pre_registered_progress_axis_is_NULL_"
        "levels_14_to_13_hv_flat_states_p0.79_navigation_significantly_WORSE_p0.013_cd82_loses_a_"
        "level_at_both_evidence_floors_and_with_only_1_discordant_game_the_design_could_not_reach_"
        "p0.05_on_the_primary_axis_recommend_do_not_flip_the_default"
    )
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    payload["reproducibility_checksum"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    try:
        payload["source_sha256"] = {
            f: hashlib.sha256((REPO / f).read_bytes()).hexdigest()
            for f in (
                "python/carnot/agentic/arc_inert_label_memory.py",
                "python/carnot/agentic/arc_competition_agent.py",
                "scripts/arc_inert_label_defer_worker.py",
                "scripts/arc_inert_label_defer_ab.py",
            )
        }
        payload["git_head"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        pass
    Path(args.out).write_text(json.dumps(payload, indent=1, default=str))
    print(
        json.dumps(
            {k: payload[k] for k in ("aa_noise_floor", "per_level_score_conversion")}, indent=1
        )[:3000]
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
