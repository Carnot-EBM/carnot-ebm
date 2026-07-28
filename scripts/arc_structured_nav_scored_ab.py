# ---------------------------------------------------------------------------------------------
# FROZEN PRE-2026-07-28 HARNESS -- MEASURES THE **RETIRED** GENERATOR.
#
# This script pins Qwen3.5-9B-MTP and describes itself as reproducing the scored/live path. That
# was true when it was written and is NO LONGER TRUE: the operator directive of 2026-07-28
# re-pinned the live ARC generator to gemma-4-31B-it (13 games x 3 replicates, fail-as-zero 0.3843
# vs 0.0627, matched 11-0-2, sign p=0.00098), and the live sites now read
# `ARC_LIVE_GENERATOR_REPO_SUBSTR` from `arc_executable_world_model`.
#
# The pin below is DELIBERATELY LEFT ALONE rather than migrated, because this harness's recorded
# historical results were genuinely taken on the 9B and re-pointing it would silently change what
# those results mean (never-prune). The cost of leaving it is that a NEW run through this file
# measures the retired generator while its own prose claims to characterise the live one.
#
# THEREFORE: results produced by this script from 2026-07-28 onward are NOT citable as live-path
# evidence. If you need a live-path A/B, read the canonical constants instead of this pin.
# ---------------------------------------------------------------------------------------------
#!/usr/bin/env python3
"""Scored-path A/B: does the structured nav inducer (CARNOT_ARC_STRUCTURED_NAV=1) improve the REAL
E3AgentPolicy cascade on a navigation game? (REQ-ARC-WMTE-5842)

This is the submission-gate evidence. It drives the SAME whole-loop scored cascade the Kaggle submission uses
(`arc_actions_to_progress.run_bounded_progress` -> E3AgentPolicy explore->stall->induce->plan->execute) on the
clean navigation game tu93, with the structured-nav flag OFF (baseline: LLM induction, which the 2026-07-20
diagnosis found near-universally wrong -- heldout ~0.0) vs ON (treatment: fit InducedNavWorldModel first, gate,
plan_in_model). Same proposer, seed, budget. Measures reached_level / levels_gained / n_plans_found /
mean_heldout_accuracy per arm.

Per the ARC submission-gating discipline (arc3-online-gated-on-offline-beating-baselines), a new SCORED
submission is only justified when an OFFLINE result beats the prior; this produces that offline evidence for
the nav-family case. tu93 is the clean navigation game; ls20 (attribute-morph win, NOT pure nav) is a negative
control -- the nav inducer's "cover the goal" win predicate should NOT match it, so it should correctly not
help there.

inference_substrate: live_llm_inference (baseline arm loads the Qwen generator; the structured-nav arm may
short-circuit before the LLM). solve_provenance: development_proxy (offline eval; no registry change).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# Broader gate set (REQ-ARC-WMTE-5843): the 3 games where the nav inducer FIRES (fits + plans a win --
# scanned no-LLM 2026-07-24: tu93/sk48/wa30) + controls where it does NOT install a plan (ls20 fits but no
# plan -> falls through; sc25 not a nav game). Override with ARC_AB_GAMES="a,b,c".
_DEFAULT_GAMES = ["tu93", "sk48", "wa30", "ls20", "sc25"]
NAV_FIRE = {"tu93", "sk48", "wa30"}  # games where the structured nav inducer installs a plan
GAMES = [g for g in os.environ.get("ARC_AB_GAMES", ",".join(_DEFAULT_GAMES)).split(",") if g]
SEED = 5842
BUDGET = 220
MAX_INDUCTIONS = 3
WALL_S = 420.0
EXPLORE_BUDGET = 24


def _run(game: str, structured_nav: bool, proposer) -> dict:
    from carnot.agentic import arc_actions_to_progress as atp

    if structured_nav:
        os.environ["CARNOT_ARC_STRUCTURED_NAV"] = "1"
    else:
        os.environ.pop("CARNOT_ARC_STRUCTURED_NAV", None)
    r = atp.run_bounded_progress(
        game,
        "frozen",
        proposer=proposer,
        seed=SEED,
        budget=BUDGET,
        max_inductions=MAX_INDUCTIONS,
        wall_s=WALL_S,
        explore_budget=EXPLORE_BUDGET,
    )
    return {
        "reached_level": r.reached_level,
        "levels_gained": r.levels_gained,
        "solved": r.solved,
        "n_inductions": r.n_inductions,
        "n_plans_found": r.n_plans_found,
        "mean_heldout_accuracy": r.mean_heldout_accuracy,
        "hv_progress": r.hv_progress,
        "total_actions": r.total_actions,
        "wall_s": r.wall_s,
    }


def main() -> int:
    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")  # outer-loop owns GPU 1
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    prop = LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP",
        port=int(os.environ.get("SNAV_PORT", "8961")),
        mtp=True,
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=4096,
        timeout=600,
    )
    t0 = time.time()
    per_game = []
    for game in GAMES:
        row = {"game": game}
        # treatment first (structured nav fires before the LLM -> often fast); then baseline.
        row["structured_nav_on"] = _run(game, True, prop)
        row["baseline_off"] = _run(game, False, prop)
        b, s = row["baseline_off"], row["structured_nav_on"]
        row["reached_delta"] = (s.get("reached_level") or 0) - (b.get("reached_level") or 0)
        per_game.append(row)
        print(
            f"[{game}] baseline reached L{b.get('reached_level')} (heldout {b.get('mean_heldout_accuracy')}, "
            f"plans {b.get('n_plans_found')}) | structured_nav reached L{s.get('reached_level')} "
            f"(heldout {s.get('mean_heldout_accuracy')}, plans {s.get('n_plans_found')}) | delta {row['reached_delta']}"
        )

    tu = next((r for r in per_game if r["game"] == "tu93"), {})
    tu_delta = tu.get("reached_delta", 0)

    # Aggregate gate metrics
    def _reached(row, arm):
        return int((row.get(arm, {}) or {}).get("reached_level", 0) or 0)

    total_off = sum(_reached(r, "baseline_off") for r in per_game)
    total_on = sum(_reached(r, "structured_nav_on") for r in per_game)
    nav_rows = [r for r in per_game if r["game"] in NAV_FIRE]
    ctrl_rows = [r for r in per_game if r["game"] not in NAV_FIRE]
    nav_improved = sum(1 for r in nav_rows if r["reached_delta"] > 0)
    nav_regressed = sum(1 for r in nav_rows if r["reached_delta"] < 0)
    ctrl_regressed = sum(1 for r in ctrl_rows if r["reached_delta"] < 0)
    ctrl_improved = sum(1 for r in ctrl_rows if r["reached_delta"] > 0)
    gate_clears = (
        total_on >= total_off and nav_improved >= 1 and ctrl_regressed == 0 and nav_regressed == 0
    )
    aggregate = {
        "n_games": len(per_game),
        "total_reached_off": total_off,
        "total_reached_on": total_on,
        "total_delta": total_on - total_off,
        "nav_games": [r["game"] for r in nav_rows],
        "nav_improved": nav_improved,
        "nav_regressed": nav_regressed,
        "control_games": [r["game"] for r in ctrl_rows],
        "control_improved": ctrl_improved,
        "control_regressed": ctrl_regressed,
        "gate_clears": bool(gate_clears),
    }
    art = {
        "experiment": "outer_loop_arc_structured_nav_broader_ab",
        "experiment_id": "REQ-ARC-WMTE-5843",
        "run_date": "2026-07-23",
        "schema": "carnot.arc_structured_nav_scored_ab.v1",
        "title": "Scored-path A/B: does CARNOT_ARC_STRUCTURED_NAV=1 improve the real E3AgentPolicy cascade on a navigation game (tu93)? Submission-gate evidence.",
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "random_seed": SEED,
        "model_specs": [
            {"name": "Qwen3.5-9B-MTP-GGUF", "gpu": 1, "role": "world_model_induction_proposer"}
        ],
        "config": {
            "budget": BUDGET,
            "max_inductions": MAX_INDUCTIONS,
            "explore_budget": EXPLORE_BUDGET,
            "wall_s": WALL_S,
        },
        "methodology_note": "Same whole-loop scored cascade (run_bounded_progress -> E3AgentPolicy) as the Kaggle submission, same proposer/seed/budget; the ONLY change is CARNOT_ARC_STRUCTURED_NAV. tu93 = clean nav (decisive); ls20 = morph-win negative control (nav inducer should not help). Public-game frames for offline dev.",
        "per_game": per_game,
        "tu93_reached_delta": tu_delta,
        "aggregate": aggregate,
        "gate_evidence": (
            "GATE CLEARS: aggregate levels improved (or held) with nav gains and NO control/nav regression"
            if gate_clears
            else "GATE DOES NOT CLEAR -- see aggregate (regression or no net gain); do NOT submit on this basis"
        ),
        "duration_s": round(time.time() - t0, 1),
    }
    art["honest_verdict"] = (
        f"complete_structured_nav_broader_ab_total_delta_{aggregate['total_delta']}_gate_clears_{gate_clears}"
    )
    art["reproducibility_checksum"] = (
        "sha256:"
        + hashlib.sha256(json.dumps(art, sort_keys=True, default=str).encode()).hexdigest()
    )
    out = ROOT / "results" / "outer_loop_arc_structured_nav_broader_ab_20260724.json"
    out.write_text(json.dumps(art, indent=2, default=str))
    print(
        f"AGGREGATE: total reached {total_off}->{total_on} (delta {aggregate['total_delta']}) | "
        f"nav improved {nav_improved}/{len(nav_rows)} | control regressed {ctrl_regressed}/{len(ctrl_rows)} | "
        f"GATE_CLEARS={gate_clears}"
    )
    print("wrote", out, f"({art['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
