#!/usr/bin/env python3
"""End-to-end goal gate: does DETECTOR-produced perception recover goal induction? (REQ-ARC-WMTE-5834).

This closes the loop opened by:
  - REQ-ARC-WMTE-5831: perception is the ARC wall (0/8 goals right under naive segmentation perception).
  - REQ-ARC-WMTE-5832: HAND-AUTHORED oracle perception flips goal induction 0/8 -> 7/8. (Proved the target.)
  - REQ-ARC-WMTE-5833: the HUD + mover detectors that PRODUCE those facts from the agent's own frames.

The decisive question this answers: when the perception fed to gemma-4-31B comes from the DETECTORS (run on
real offline-arcade transitions) instead of a human reading source, does its goal hypothesis still flip to
the true win condition? Same model, same learned RULES per game as REQ-ARC-WMTE-5832 (held constant -- only
the perception source changes: hand-authored oracle -> detector output). Target: recover the ~7/8
goal-correctness. A shortfall is honest data about which detector output is still too thin (e.g. the detector
lists candidate target objects but cannot NAME which is the exit -- gemma must infer that).

inference_substrate: live_llm_inference (gemma-4-31B GGUF on GPU). Detectors + arcade use no LLM; only the
goal-hypothesis step does. Public-game frames stepped for offline dev validation (authorized), never in the
hidden submission.
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

    from carnot.agentic.arc_entity_hud_perception import perceive_entities

    port = int(os.environ.get("E2E_PORT", "8958"))
    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")  # outer-loop owns GPU 1
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    t0 = time.time()
    proposer = LocalGGUFProposer(
        repo_substr="gemma-4-31B-it",
        port=port,
        mtp=False,
        kv_quant="q8_0",
        n_ctx=8192,
        max_tokens=256,
        no_think_prefix="",
    )
    seeds = [5834, 5835]
    per_game = []
    for game, spec in GAMES.items():
        try:
            trans, shape, lvl = _gather(game)
        except Exception as exc:  # noqa: BLE001
            per_game.append({"game": game, "error": repr(exc)[:200]})
            print(f"[{game}] gather ERROR: {exc!r}")
            continue
        percept = perceive_entities(trans[-1].after, trans) if trans else None
        percept_text = percept.text if percept else "(no perception)"
        goals = [_goal(proposer, spec["rules"], percept_text, s) for s in seeds]
        per_game.append(
            {
                "game": game,
                "true_win": spec["true_win"],
                "perception_blocked": spec["perception_blocked"],
                "n_transitions": len(trans),
                "detector_perception": percept_text,
                "detector_hud": [f"{b.axis}{b.index}/{b.direction}" for b in (percept.hud_bands if percept else [])],
                "detector_mover": None if not percept or percept.mover is None else percept.mover.color,
                "goals_under_detector_perception": goals,
            }
        )
        print(f"[{game}] mover={per_game[-1]['detector_mover']} hud={per_game[-1]['detector_hud']}")
        print(f"   true_win: {spec['true_win']}")
        for i, g in enumerate(goals):
            print(f"   goal[{i}]: {g}")
        print()

    art = {
        "experiment": "outer_loop_arc_end_to_end_goal_gate",
        "experiment_id": "REQ-ARC-WMTE-5834",
        "run_date": "2026-07-23",
        "schema": "carnot.arc_end_to_end_goal_gate.v1",
        "title": "Does DETECTOR-produced perception (HUD + mover, from real arcade frames) recover gemma-4-31B's goal induction, matching the hand-authored oracle's 7/8?",
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "random_seed": 5834,
        "random_seeds_used": seeds,
        "model_specs": [
            {"name": "gemma-4-31B-it-GGUF", "repo": "unsloth/gemma-4-31B-it-GGUF", "gpu": 1, "port": port}
        ],
        "methodology_note": "Same model + same learned RULES per game as REQ-ARC-WMTE-5832; the ONLY change is the perception SOURCE -- hand-authored oracle facts (5832) replaced by perceive_entities() output run on real offline-arcade transitions (5833 detectors). Compares detector perception to the oracle's 7/8. Detectors + arcade use no LLM. Public-game frames stepped for offline dev validation (authorized), never in the hidden submission.",
        "baseline_oracle_result": "REQ-ARC-WMTE-5832 hand-authored oracle: 7/8 correct + 1 partial, 0 wrong (naive was 0/8).",
        "per_game": per_game,
        "duration_s": round(time.time() - t0, 2),
    }
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(json.dumps(art, sort_keys=True).encode()).hexdigest()
    out = ROOT / "results" / "outer_loop_arc_end_to_end_goal_gate_20260723.json"
    out.write_text(json.dumps(art, indent=2))
    print("wrote", out, f"({art['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
