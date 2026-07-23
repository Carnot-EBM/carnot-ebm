#!/usr/bin/env python3
"""Re-authored end-to-end goal gate (REQ-ARC-WMTE-5834 fix).

The plain end-to-end gate (REQ-ARC-WMTE-5834) fed detector perception ALONGSIDE the model's own WRONG learned
rules and did NOT recover the oracle's 7/8 (0 correct, 2 partial, 6 wrong): the model followed a rule it
already believed even when the detector flagged that band as a counter. This variant applies
`reauthor_framing` -- it RETRACTS any learned rule that references a detected HUD band, NAMES the mover's
nearest object as the candidate target, and OVERRIDES the counter-fixation -- so the correction REPLACES the
wrong framing rather than sitting beside it (what the hand-authored oracle did to get 7/8). Everything else is
held identical to REQ-ARC-WMTE-5834 (same model, same games, same gather, same seeds), so the delta isolates
re-authoring.

inference_substrate: live_llm_inference (gemma-4-31B GGUF on GPU). Detectors + arcade use no LLM.
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
sys.path.insert(0, str(ROOT))


def main() -> int:
    from scripts.arc_oracle_perception_goal_ablation import GAMES, _goal
    from scripts.arc_perception_detectors_realframe_gate import _gather

    from carnot.agentic.arc_entity_hud_perception import perceive_entities, reauthor_framing

    port = int(os.environ.get("E2E_PORT", "8959"))
    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    t0 = time.time()
    proposer = LocalGGUFProposer(
        repo_substr="gemma-4-31B-it", port=port, mtp=False, kv_quant="q8_0",
        n_ctx=8192, max_tokens=256, no_think_prefix="",
    )
    seeds = [5834, 5835]  # SAME seeds as the plain gate, so the only change is re-authoring
    per_game = []
    for game, spec in GAMES.items():
        try:
            trans, shape, lvl = _gather(game)
        except Exception as exc:  # noqa: BLE001
            per_game.append({"game": game, "error": repr(exc)[:200]})
            print(f"[{game}] gather ERROR: {exc!r}")
            continue
        percept = perceive_entities(trans[-1].after, trans) if trans else None
        if percept is None:
            corrected_rules, perception_block = spec["rules"], "(no perception)"
        else:
            corrected_rules, perception_block = reauthor_framing(spec["rules"], percept)
        goals = [_goal(proposer, corrected_rules, perception_block, s) for s in seeds]
        per_game.append(
            {
                "game": game,
                "true_win": spec["true_win"],
                "n_transitions": len(trans),
                "detector_mover": None if not percept or percept.mover is None else percept.mover.color,
                "detector_hud": [f"{b.axis}{b.index}/{b.direction}" for b in (percept.hud_bands if percept else [])],
                "corrected_rules": corrected_rules,
                "perception_block": perception_block,
                "goals_under_reauthored_perception": goals,
            }
        )
        print(f"[{game}] mover={per_game[-1]['detector_mover']} hud={per_game[-1]['detector_hud']}")
        print(f"   true_win: {spec['true_win']}")
        for i, g in enumerate(goals):
            print(f"   goal[{i}]: {g}")
        print()

    art = {
        "experiment": "outer_loop_arc_end_to_end_goal_gate_reauthored",
        "experiment_id": "REQ-ARC-WMTE-5834",
        "run_date": "2026-07-23",
        "schema": "carnot.arc_end_to_end_goal_gate_reauthored.v1",
        "title": "Does RE-AUTHORED detector perception (retract HUD rules + name target + override framing) recover gemma-4-31B goal induction toward the oracle's 7/8?",
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "random_seed": 5834,
        "random_seeds_used": seeds,
        "model_specs": [{"name": "gemma-4-31B-it-GGUF", "repo": "unsloth/gemma-4-31B-it-GGUF", "gpu": 1, "port": port}],
        "methodology_note": "Identical to REQ-ARC-WMTE-5834 (same model/games/gather/seeds) EXCEPT reauthor_framing replaces the plain perceive_entities text: retract HUD-band rules, name the mover's nearest object as target, override the counter-fixation. Isolates the re-authoring delta. Public-game frames stepped for offline dev validation (authorized), never in the hidden submission.",
        "baseline_plain_gate": "REQ-ARC-WMTE-5834 plain: 0 correct, 2 partial, 6 wrong. Oracle (5832): 7/8 correct.",
        "per_game": per_game,
        "duration_s": round(time.time() - t0, 2),
    }
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(json.dumps(art, sort_keys=True).encode()).hexdigest()
    out = ROOT / "results" / "outer_loop_arc_end_to_end_goal_gate_reauthored_20260723.json"
    out.write_text(json.dumps(art, indent=2))
    print("wrote", out, f"({art['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
