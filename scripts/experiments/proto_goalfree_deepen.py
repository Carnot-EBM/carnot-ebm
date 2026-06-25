#!/usr/bin/env python3
"""Goal-free L1->L2 deepening probe (the leader's mechanism), measured on the SCORED E3AgentPolicy.

WHY (operator-directed 2026-06-25, the next multi-level iterative step)
----------------------------------------------------------------------
The .430 goal-predicate FIX (win-state exemplar + satisfiability check) shipped, ran (exp4664), and
NULLED+RETIRED (single_exemplar_goal_insufficient; goal_predicate_satisfiable=False; the live
multi-level solve rate is 0.0 and every banked level is a development_proxy adapter). So
"fix the goal predicate" is a doomed rerun. The 2026-06-24 leader-gap analysis reframed the wall:
StochasticGoose (leaderboard #1) deepens multi-level FOR FREE with NO goal predicate -- reward-driven
systematic exploration + per-level reset. Carnot ALREADY HAS that machinery (GoExploreReplayArchive,
return-then-explore) wired into StepwiseExplorer but DISABLED in the scored path
(SUBMITTED_GO_EXPLORE_ARCHIVE_ENABLED=False). This probe is the cheap, decisive disambiguator: does
GOAL-FREE systematic exploration (Go-Explore coverage), riding past L1, deepen lp85 to L2 where the
default explorer does not -- BEFORE committing to the expensive CNN-as-DRIVER build.

DESIGN (parity-safe; NO shipped-code edit)
------------------------------------------
We monkeypatch mod._policy_for_mode (exp4605 harness) to build the integrated policy with
go_explore_archive=<arm> and target_levels=2, then run mod.run_variant_attempt with DEEPEN=1 (ride
past the first level-up) + DISABLE_INDUCTION=1 (skip the LLM goal-induction tier entirely -> a PURE
goal-free loop). The exp4605 NoOpProposer keeps it CPU-only (no Qwen). Two arms isolate the Go-Explore
archive's effect:
  no_archive : goal-free explorer, archive OFF  (control -- does lp85 deepen by default coverage?)
  go_explore : goal-free explorer, archive ON   (the lever -- does systematic return-then-explore deepen?)

GATE: go_explore reaches L2 (depth_reached>=2), offline-reproduced via kit.reproduce, where no_archive
does not -> goal-free deepening validated, the CNN driver is justified. If BOTH stay at L1 -> goal-free
coverage alone is insufficient; redirect to the multi-exemplar goal-fix variant (NOT the CNN driver).
solve_provenance=live_agent_self_discovery (the SCORED E3AgentPolicy, its own exploration; NOT an adapter).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

# Pure goal-free loop: ride past L1, skip the LLM goal-induction tier entirely.
os.environ["CARNOT_ARC_GATE_DEEPEN"] = "1"
os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"

import carnot.experiment_4605_live_integration_scored_agent as mod  # noqa: E402
from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: E402

TARGET_GAME = "lp85"
VARIANTS = [1, 2, 3]
BUDGET = 1200  # deepening needs a long goal-free horizon; also long enough to EXERCISE the archive
ARMS = ["no_archive", "go_explore"]
# Finer-than-default archive (bins 16 vs 6, more cells) so the return-then-explore replay actually
# FIRES (the default 6-bin archive stored only 2 cells -> 0 injections -> the lever was untested).
GO_EXPLORE_CONFIG = {"bins": 16, "max_cells": 512}


def _run_arm(arm: str) -> dict:
    built: list = []  # capture policies for go-explore diagnostics

    def patched_policy_for_mode(mode: str, game: str):
        if mode == "bare":
            return mod._policy_for_mode_orig(mode, game)
        pol = E3AgentPolicy(
            game,
            proposer=mod._NoOpProposer(),
            target_levels=2,  # ride L1 -> L2
            value_weight=mod._submitted_value_weight(),
            go_explore_archive=(GO_EXPLORE_CONFIG if arm == "go_explore" else False),
        )
        built.append(pol)
        return pol

    mod._policy_for_mode_orig = getattr(mod, "_policy_for_mode_orig", mod._policy_for_mode)
    mod._policy_for_mode = patched_policy_for_mode
    rows = []
    try:
        for v in VARIANTS:
            spec = mod.variant_specs([TARGET_GAME], [v])[0]
            row = dict(mod.run_variant_attempt("integrated", TARGET_GAME, spec, BUDGET))
            rows.append(row)
    finally:
        mod._policy_for_mode = mod._policy_for_mode_orig

    # Go-Explore diagnostics from the built policies (prefixes/actions injected proves the archive ran).
    ge = {"prefixes_injected": 0, "actions_injected": 0, "archive_active": False}
    for pol in built:
        try:
            d = pol.explorer.go_explore_archive_diagnostics()
            if d:
                ge["archive_active"] = True
                ge["prefixes_injected"] += int(d.get("prefixes_injected") or 0)
                ge["actions_injected"] += int(d.get("actions_injected") or 0)
        except Exception:
            pass

    depths = [int(r.get("depth_reached") or 0) for r in rows]
    reached = [int(r.get("reached_level") or 0) for r in rows]
    l2_rows = [r for r in rows if int(r.get("depth_reached") or 0) >= 2 and r.get("solved")]
    return {
        "arm": arm,
        "per_variant": [
            {
                "variant": r["variant"],
                "reached_level": int(r.get("reached_level") or 0),
                "depth_reached": int(r.get("depth_reached") or 0),
                "solved": bool(r.get("solved")),
                "actions": int(r.get("actions") or 0),
                "reproduced": bool(r.get("reproduction_gate", {}).get("reproduced")),
            }
            for r in rows
        ],
        "max_depth_reached": max(depths) if depths else 0,
        "max_level_reached": max(reached) if reached else 0,
        "n_variants_reached_l1": sum(1 for x in reached if x >= 1),
        "n_variants_reached_l2": len(l2_rows),
        "go_explore": ge,
    }


def main() -> int:
    started = time.time()
    results = {arm: _run_arm(arm) for arm in ARMS}

    base = results["no_archive"]
    lever = results["go_explore"]
    # Genuine goal-free L2: go_explore reaches an offline-reproduced L2 where no_archive does not.
    go_l2 = lever["n_variants_reached_l2"] >= 1
    base_l2 = base["n_variants_reached_l2"] >= 1
    crossed = bool(go_l2 and not base_l2)
    # Positive control: the probe must actually reach L1 (else "no L2" is uninformative).
    reached_l1 = lever["n_variants_reached_l1"] >= 1 or base["n_variants_reached_l1"] >= 1
    # The Go-Explore lever is only EXERCISED if it actually injected a return prefix (prefixes
    # injected > 0). A 0-injection archive is a passive no-op -> a null would be a false negative
    # (the lever was never tested), exactly the trap the exp4710 false negative turned on.
    archive_ran = int(lever["go_explore"].get("prefixes_injected") or 0) > 0

    if crossed:
        verdict = (
            f"success: goal_free_go_explore_deepens_{TARGET_GAME}_to_l2 "
            f"go_explore_l2={lever['n_variants_reached_l2']} baseline_l2={base['n_variants_reached_l2']}"
        )
    elif go_l2 and base_l2:
        verdict = "complete: lp85_deepens_to_l2_without_archive_too_archive_not_load_bearing"
    elif not reached_l1:
        verdict = "complete: goal_free_deepen_no_l2_residual_never_reached_l1_uninformative"
    elif not archive_ran:
        verdict = "complete: goal_free_deepen_no_l2_residual_archive_not_exercised_false_negative_risk"
    else:
        # Valid null: L1 reached AND the Go-Explore archive actually injected return prefixes, yet
        # no L2 -> goal-free systematic exploration is insufficient to deepen lp85; redirect to the
        # multi-exemplar goal-fix, NOT the CNN driver.
        verdict = "complete: goal_free_deepen_no_l2_valid_null_coverage_insufficient_redirect_multi_exemplar_goalfix"

    artifact = {
        "experiment": "proto_goalfree_deepen",
        "schema": "carnot.proto.goalfree_deepen.v1",
        "target_game": TARGET_GAME,
        "variants": VARIANTS,
        "budget": BUDGET,
        "deepening_mechanism": "goal_free_go_explore",
        "arms": results,
        "go_explore_archive_reached_l2": go_l2,
        "baseline_reached_l2": base_l2,
        "crossed_bar": crossed,
        "reached_l1_positive_control": reached_l1,
        "archive_actually_ran": archive_ran,
        "honest_verdict": verdict,
        "solve_provenance": "live_agent_self_discovery",
        "verifier_is_oracle": False,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "random_seed": 4720,
        "duration_s": round(max(1.0, time.time() - started), 3),
        "field_principles": {
            "go_explore_archive_reached_l2": "the lever: did goal-free systematic exploration deepen past L1",
            "crossed_bar": "go_explore reaches an offline-reproduced L2 where the no-archive control does not",
            "reached_l1_positive_control": "the probe must reach L1, else 'no L2' is an uninformative false negative",
            "archive_actually_ran": "prefixes/actions injected > 0 proves the Go-Explore archive was actually exercised, not a no-op",
        },
    }
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    artifact["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()

    out = REPO / "results/proto_goalfree_deepen.json"
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")

    print(f"\n{'arm':12} {'maxLvl':>7} {'maxDepth':>9} {'L1/3':>5} {'L2/3':>5} {'archive(pfx/act)'}")
    for arm in ARMS:
        r = results[arm]
        ge = r["go_explore"]
        print(
            f"{arm:12} {r['max_level_reached']:>7} {r['max_depth_reached']:>9} "
            f"{r['n_variants_reached_l1']:>5} {r['n_variants_reached_l2']:>5} "
            f"{ge['prefixes_injected']}/{ge['actions_injected']} active={ge['archive_active']}"
        )
    print(f"\ncrossed_bar={crossed}  reached_l1_control={reached_l1}  archive_ran={archive_ran}")
    print(f"VERDICT: {verdict}")
    print(f"written: {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
