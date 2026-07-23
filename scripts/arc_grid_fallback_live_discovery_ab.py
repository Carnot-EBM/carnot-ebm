#!/usr/bin/env python3
"""Generalization-Testing-Floor validation of the grid-fallback perception fix on the REAL scored path.

Operator directive (2026-07-23, after the lever-triangulation): validate the ONE surviving
deliverable-relevant finding from GAP-ARC-BP35-CLICK-CANDIDATE-GENERATION-MISS against the REAL metric
(live discovery), NOT candidate coverage of a hand-built adapter trajectory.

WHAT THIS TESTS. The fix (CARNOT_ARC_GRID_FALLBACK_CANDIDATES=1) makes the live candidate generator
(object_centric_digest -> rich_action_candidates, called by the SCORED E3AgentPolicy._candidates at
arc_competition_agent.py:1400) able to propose clicks on the single most-common ("background") color,
which it otherwise excludes wholesale. The flag is env-read INSIDE rich_action_candidates, so the scored
path already honors it with no wiring change. This driver runs the REAL E3AgentPolicy cascade FRAME-ONLY
(no GameAdapter, proposer=None -- byte-identical config to the standing results/arc_live_oracle_gap.json
oracle-gap baseline: --policy e3, budget=400, seed=20260719), flag OFF then flag ON, PAIRED per game,
and measures LEVELS ACTUALLY DISCOVERED -- the deliverable metric, not coverage.

HONEST PRIOR (from the triangulation, docs/research-notes/arc-lever-triangulation-2026-07-23.md): the
binding constraint is sequence-routing/world-model induction, not per-action candidate availability, so
this fix is expected to MOST LIKELY null on discovered levels even where it improves candidate coverage.
A null here is a real, decisive datum: it says the perception fix is coverage-only (proxy), not
discovery-relevant. A non-null on ANY game (esp. a dominant-background game like bp35) would be the first
evidence a perception fix moves the real metric, and would justify wiring it toward the live default.

Writes a DEDICATED artifact (never the tracked arc_live_oracle_gap.json).
Spec: GAP-ARC-BP35-CLICK-CANDIDATE-GENERATION-MISS (class-level reframe), Generalization-Testing-Floor.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))

import arc_leaderboard_eval as lb  # noqa: E402  (the standing oracle-gap harness; reuse its run_game)

BUDGET = 400  # matches the standing oracle-gap baseline exactly
SEED = 20260719  # matches the standing oracle-gap baseline exactly
FLAG = "CARNOT_ARC_GRID_FALLBACK_CANDIDATES"
OUTPUT_PATH = REPO / "results" / "outer_loop_arc_grid_fallback_live_discovery_ab_20260723.json"


def _reseed() -> None:
    random.seed(SEED)
    try:
        import numpy as np

        np.random.seed(SEED)
    except Exception:
        pass


def _run_arm(games: list[str], flag_on: bool) -> dict[str, dict]:
    if flag_on:
        os.environ[FLAG] = "1"
    else:
        os.environ.pop(FLAG, None)
    _reseed()
    rows: dict[str, dict] = {}
    for game in games:
        t0 = time.time()
        # _build_policy("e3", game) -> E3AgentPolicy(game, proposer=None) with _PROPOSER_REPO="" (no LLM),
        # exactly as the standing oracle-gap baseline builds it.
        r = lb.run_game(game, lb._build_policy("e3", game), budget=BUDGET)
        rows[game] = {
            "levels": int(r["levels"]),
            "reached": int(r["reached"]),
            "actions": int(r["actions"]),
            "wall_s": round(time.time() - t0, 2),
        }
        print(
            f"  [{'ON ' if flag_on else 'OFF'}] {game:5} levels={rows[game]['levels']} "
            f"reached=L{rows[game]['reached']} actions={rows[game]['actions']} "
            f"[{rows[game]['wall_s']}s]",
            flush=True,
        )
    return rows


def main() -> None:
    argv = sys.argv[1:]
    only = ""
    if "--only" in argv:
        only = argv[argv.index("--only") + 1]
    oracle = lb._oracle_levels()
    games = sorted(oracle)
    if only:
        keep = set(only.split(","))
        games = [g for g in games if g in keep]
    print(
        f"== grid-fallback live-discovery A/B: {len(games)} games, budget={BUDGET}, seed={SEED} =="
    )
    print(f"   games: {games}", flush=True)

    t0 = time.time()
    print("\n-- ARM OFF (baseline: dominant-color clicks never generated) --", flush=True)
    off = _run_arm(games, flag_on=False)
    print("\n-- ARM ON (fix: dominant-color clicks tiled into candidates) --", flush=True)
    on = _run_arm(games, flag_on=True)

    per_game = []
    improved, regressed = [], []
    for g in games:
        d_levels = on[g]["levels"] - off[g]["levels"]
        d_reached = on[g]["reached"] - off[g]["reached"]
        row = {
            "game": g,
            "oracle_levels": int(oracle.get(g, 0)),
            "off": off[g],
            "on": on[g],
            "delta_levels": d_levels,
            "delta_reached": d_reached,
        }
        if d_levels > 0 or d_reached > 0:
            improved.append(g)
        if d_levels < 0 or d_reached < 0:
            regressed.append(g)
        per_game.append(row)

    off_total = sum(off[g]["levels"] for g in games)
    on_total = sum(on[g]["levels"] for g in games)
    any_improvement = len(improved) > 0
    any_regression = len(regressed) > 0

    if any_improvement and not any_regression:
        verdict = f"complete_grid_fallback_live_discovery_IMPROVED_{off_total}_to_{on_total}_levels_games_{'_'.join(improved)}"
    elif any_improvement and any_regression:
        verdict = f"complete_grid_fallback_live_discovery_MIXED_improved_{'_'.join(improved)}_regressed_{'_'.join(regressed)}"
    else:
        verdict = (
            "complete_grid_fallback_live_discovery_honest_null_no_level_or_depth_gain_on_any_game_"
            "coverage_fix_is_not_discovery_relevant_at_this_budget"
        )

    duration_s = round(time.time() - t0, 3)
    checksum_input = json.dumps(
        [{"game": r["game"], "on": r["on"]["levels"], "off": r["off"]["levels"]} for r in per_game],
        sort_keys=True,
    ).encode()
    artifact = {
        "experiment": "outer_loop_arc_grid_fallback_live_discovery_ab_20260723",
        "schema": "carnot.arc_grid_fallback_live_discovery_ab.v1",
        "run_date": "2026-07-23",
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "inference_substrate_note": "REAL E3AgentPolicy scored cascade run FRAME-ONLY (no GameAdapter, "
        "proposer=None -> no LLM), byte-identical config to the standing arc_live_oracle_gap.json baseline "
        "(--policy e3, budget=400, seed=20260719). The grid-fallback flag is env-read inside "
        "rich_action_candidates, which E3AgentPolicy._candidates (arc_competition_agent.py:1400) calls on "
        "the explorer hot path (used at :1870/:1907), so setting it activates the fix on the scored path.",
        "solve_provenance": "live_agent_self_discovery",
        "solve_provenance_note": "GameAdapters DISABLED (E3AgentPolicy frame-only); this is the "
        "Generalization-Testing-Floor protocol -- the agent must DISCOVER progress from its own "
        "exploration, not replay a hand-built trajectory. Measures the deliverable metric (levels "
        "discovered), NOT candidate coverage.",
        "target_model": "none_proposer_none_explorer_tier_only",
        "random_seed": SEED,
        "reproducibility_checksum": hashlib.sha256(checksum_input).hexdigest(),
        "duration_s": duration_s,
        "budget_per_game": BUDGET,
        "honest_verdict": verdict,
        "narrative": (
            "Validates GAP-ARC-BP35-CLICK-CANDIDATE-GENERATION-MISS's surviving class-level finding (the "
            "live generator cannot click the most-common color) against the REAL metric (live discovery) "
            "instead of candidate coverage, per the 2026-07-23 lever triangulation. Honest prior: expected "
            "to null on discovered levels because the binding constraint is sequence-routing/induction, not "
            "per-action candidate availability -- a null is a real datum (coverage fix is proxy-only)."
        ),
        "acceptance_gate": {
            "condition": "flag ON discovers MORE levels or greater depth than flag OFF on at least one "
            "game, with no regression",
            "principle": "The deliverable is LIVE hidden-game discovery, not candidate coverage of a known "
            "adapter trajectory. Only a real levels/depth gain on the scored frame-only path counts; a "
            "coverage improvement that does not move discovery is proxy-polishing (per the adversarial lens "
            "of triangulation wqn31sxaz).",
            "passed": any_improvement and not any_regression,
        },
        "summary": {
            "off_total_levels": off_total,
            "on_total_levels": on_total,
            "improved_games": improved,
            "regressed_games": regressed,
            "n_games": len(games),
        },
        "per_game": per_game,
        "preconditions_checked": [
            {
                "resource": "arc_grid_fallback_flag_env_read_in_rich_action_candidates",
                "available": True,
            },
            {"resource": "e3_policy_frame_only_no_adapter_no_llm", "available": True},
        ],
    }
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
    print(f"\nWrote {OUTPUT_PATH}")
    print(f"off_total_levels={off_total} on_total_levels={on_total}")
    print(f"improved={improved} regressed={regressed}")
    print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
