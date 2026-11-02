#!/usr/bin/env python3
"""Does re-authored perception convert to actual DISCOVERY? A/B over 8 games (REQ-ARC-WMTE-5834).

The offline goal gate showed re-authoring flips goal-HYPOTHESIS correctness 0/8 -> 4/8. The decisive question
is whether that converts to LEVELS DISCOVERED by the live greedy-direct agent. This A/B runs the full winner
recipe (gemma-4-31B, greedy-direct, object segmentation, reflection memory, goal verifier) with reauthor OFF
vs ON, same budget/seed per game, and measures `levels_gained` (real env level-ups). Baseline recall: the v5
winner-recipe sweep discovered 0 levels on all 8.

Honest expectation: right goal is necessary but not sufficient -- execution/dynamics is a further gap, so a
navigate game (tu93/sc25) with the right goal may or may not be reachable within budget, and merge/gravity
games likely still need the dynamics half. Any delta (or a clean null) is informative.

inference_substrate: live_llm_inference (gemma-4-31B GGUF on GPU). Public-game frames stepped for offline dev
validation (authorized), never in the hidden submission.
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

GAMES = ["bp35", "lf52", "sc25", "ls20", "tu93", "cn04", "r11l", "ft09"]


def main() -> int:
    from carnot.agentic.arc_greedy_direct_agent import run_greedy_direct

    budget = int(os.environ.get("DISCOVERY_BUDGET", "120"))
    port = int(os.environ.get("DISCOVERY_PORT", "8958"))
    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    proposer = LocalGGUFProposer(
        repo_substr="gemma-4-31B-it", port=port, mtp=False, kv_quant="q8_0",
        n_ctx=8192, max_tokens=256, no_think_prefix="",
    )
    t0 = time.time()
    per_game = []
    for game in GAMES:
        row = {"game": game}
        for reauthor in (False, True):
            key = "reauthor" if reauthor else "baseline"
            try:
                r = run_greedy_direct(
                    game, proposer, action_budget=budget, perception="objects",
                    reflection_interval=10, goal_verify=True, reauthor=reauthor, seed=5834,
                )
                row[key] = {
                    "levels_gained": r.levels_gained,
                    "reached_level": r.reached_level,
                    "actions_taken": r.actions_taken,
                    "final_notes": r.final_notes[:400],
                }
            except Exception as exc:  # noqa: BLE001 -- a per-run failure is a datum, not a crash
                row[key] = {"error": repr(exc)[:200]}
            print(f"[{game}] {key}: {row[key].get('levels_gained', row[key].get('error'))}")
        per_game.append(row)

    def _tot(key: str) -> int:
        return sum(int(r.get(key, {}).get("levels_gained", 0) or 0) for r in per_game)

    art = {
        "experiment": "outer_loop_arc_reauthor_discovery_sweep",
        "experiment_id": "REQ-ARC-WMTE-5834",
        "run_date": "2026-07-23",
        "schema": "carnot.arc_reauthor_discovery_sweep.v1",
        "title": "Does re-authored perception convert the winner recipe's 0-discovery to actual levels? A/B (reauthor off vs on), 8 games.",
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "random_seed": 5834,
        "model_specs": [{"name": "gemma-4-31B-it-GGUF", "repo": "unsloth/gemma-4-31B-it-GGUF", "gpu": 1, "port": port}],
        "action_budget": budget,
        "methodology_note": "Full winner recipe (greedy-direct + object seg + reflection + goal verifier), reauthor OFF vs ON, same budget/seed per game. Measures real-env levels_gained. Baseline: v5 winner sweep = 0 discovery on all 8. Public-game frames stepped for offline dev validation (authorized), never in the hidden submission.",
        "totals": {"baseline_levels": _tot("baseline"), "reauthor_levels": _tot("reauthor")},
        "per_game": per_game,
        "duration_s": round(time.time() - t0, 1),
    }
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(json.dumps(art, sort_keys=True).encode()).hexdigest()
    out = ROOT / "results" / "outer_loop_arc_reauthor_discovery_sweep_20260723.json"
    out.write_text(json.dumps(art, indent=2))
    print(f"TOTALS: baseline={art['totals']['baseline_levels']} reauthor={art['totals']['reauthor_levels']}")
    print("wrote", out, f"({art['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
