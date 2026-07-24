#!/usr/bin/env python3
"""LEVER #2 A/B (REQ-ARC-WMTE-5593-4): does goal-exemplar grading let the E3 agent bank a level it
otherwise could not, by vetoing a WRONG induced win-predicate at a deepening boundary?

Runs E3AgentPolicy offline (via arc_leaderboard_eval.run_game) on bp35/lf52/sc25 -- the exp5829 games
whose nulls were traced to a wrong induced GOAL (correct dynamics, un-graded win-predicate). Two arms
toggle ONLY CARNOT_ARC_GOAL_EXEMPLAR_GRADING; same generator, seed, budget. Metric = offline levels banked.

GATE (pre-registered, scoping plan): treatment.levels > baseline.levels for >=1 of {bp35,lf52,sc25} AND
NO game regresses (treatment.levels >= baseline.levels for all three). Else HONEST-NEGATIVE (keep flag off).

inference_substrate: live_llm_inference. verifier_is_oracle: False (the induced win-predicate never reads
the level counter). solve_provenance: development_proxy. NEVER submits. GPU: reuses the running server on
--port (default 8921); if unhealthy -> blocked_*.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

GAMES = [g for g in (os.environ.get("L2AB_GAMES") or "bp35,lf52,sc25").split(",") if g]
PORT = int(os.environ.get("L2AB_PORT", "8921"))
BUDGET = int(os.environ.get("L2AB_BUDGET", "600"))
EXPLORE_BUDGET = int(os.environ.get("L2AB_EXPLORE", "40"))
OUT = ROOT / "results" / "experiment_5832_goal_exemplar_grading_ab.json"


def _server_healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> int:
    t0 = time.time()
    pre = [{"resource": f"llama_server_port_{PORT}", "available": _server_healthy(PORT)}]
    if not pre[0]["available"]:
        art = {
            "experiment": "experiment_5832_goal_exemplar_grading_ab",
            "experiment_id": "REQ-ARC-WMTE-5593-4",
            "honest_verdict": f"complete: blocked_no_generator_server_on_port_{PORT}",
            "inference_substrate": "live_llm_inference",
            "preconditions_checked": pre,
            "duration_s": round(time.time() - t0, 2),
        }
        OUT.write_text(json.dumps(art, indent=2))
        print("BLOCKED: no healthy generator on port", PORT)
        return 0

    import arc_leaderboard_eval as lb  # noqa: E402
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    def run_arm(game: str, enabled: bool) -> int:
        os.environ["CARNOT_ARC_GOAL_EXEMPLAR_GRADING"] = "1" if enabled else "0"
        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.5-9B-MTP", port=PORT, mtp=True, kv_quant="q8_0",
            no_think_prefix="/no_think\n", max_tokens=2560, timeout=600,
        )
        policy = E3AgentPolicy(game, proposer=proposer, explore_budget=EXPLORE_BUDGET)
        r = lb.run_game(game, policy, budget=BUDGET)
        return int(r.get("levels", 0))

    per_game = []
    for game in GAMES:
        row: dict = {"game": game}
        try:
            row["baseline_levels"] = run_arm(game, False)
        except Exception as e:
            row["baseline_levels"] = None
            row["baseline_error"] = f"{type(e).__name__}: {e}"[:200]
        try:
            row["treatment_levels"] = run_arm(game, True)
        except Exception as e:
            row["treatment_levels"] = None
            row["treatment_error"] = f"{type(e).__name__}: {e}"[:200]
        b, t = row.get("baseline_levels"), row.get("treatment_levels")
        row["levels_delta"] = (t - b) if isinstance(b, int) and isinstance(t, int) else None
        per_game.append(row)
        print(f"[{game}] baseline={b} treatment={t} delta={row['levels_delta']}")
    os.environ.pop("CARNOT_ARC_GOAL_EXEMPLAR_GRADING", None)

    valid = [r for r in per_game if isinstance(r.get("levels_delta"), int)]
    improved = [r["game"] for r in valid if r["levels_delta"] > 0]
    regressed = [r["game"] for r in valid if r["levels_delta"] < 0]
    gate_clears = bool(len(improved) >= 1 and len(regressed) == 0 and len(valid) == len(per_game))
    seed = 5832
    art = {
        "experiment": "experiment_5832_goal_exemplar_grading_ab",
        "experiment_id": "REQ-ARC-WMTE-5593-4",
        "run_date": "2026-07-24",
        "title": "Lever #2 A/B: goal-exemplar grading vs off; offline E3 levels banked on bp35/lf52/sc25.",
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "random_seed": seed,
        "model_specs": [{"name": "Qwen3.5-9B-MTP-GGUF", "port": PORT, "role": "e3_induction_generator"}],
        "config": {"games": GAMES, "budget": BUDGET, "explore_budget": EXPLORE_BUDGET},
        "methodology_note": (
            "Two arms toggle ONLY CARNOT_ARC_GOAL_EXEMPLAR_GRADING; same live Qwen3.5-9B-MTP server, same "
            "budget/explore_budget per game. E3AgentPolicy offline via arc_leaderboard_eval.run_game; metric "
            "= offline levels banked. Lever only bites at L2+ (needs a prior banked level), matching where "
            "GAP-ARCH-GOAL-NOT-VERIFIED lives."
        ),
        "per_game": per_game,
        "games_improved": improved,
        "games_regressed": regressed,
        "gate_clears": gate_clears,
        "preconditions_checked": pre,
        "duration_s": round(time.time() - t0, 1),
    }
    art["honest_verdict"] = (
        f"complete_goal_exemplar_grading_ab_improved_{len(improved)}_regressed_{len(regressed)}"
        f"_gate_clears_{gate_clears}"
    )
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(art, sort_keys=True, default=str).encode()
    ).hexdigest()
    OUT.write_text(json.dumps(art, indent=2, default=str))
    print(
        f"\nAGGREGATE improved={improved} regressed={regressed} GATE_CLEARS={gate_clears} "
        f"({art['duration_s']}s) -> {OUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
