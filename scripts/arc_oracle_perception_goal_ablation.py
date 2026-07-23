#!/usr/bin/env python3
"""Oracle-perception goal-hypothesis ablation (REQ-ARC-WMTE-5832).

WHY THIS EXISTS. The winner-recipe reproduction (REQ-ARC-WMTE-5829/5830) + the source-grounded
diagnosis (REQ-ARC-WMTE-5831) found that gemma-4-31B NEVER hypothesized the true win condition on
any of 8 games -- 5/8 because its perception fixated on the HUD/budget-bar and never REPRESENTED the
player/exit/target entities. The named next lever is a perception fix (HUD-vs-board disambiguation +
mover-detection). Before BUILDING those detectors, this ablation tests the decisive counterfactual:

    If we HAND the same model the correct entities (what a perfect detector would output), does its
    GOAL hypothesis flip from wrong to right?

If yes on the perception-blocked games -> the perception fix is confirmed as THE lever and building
the detectors is justified. If no -> perception alone is insufficient and we've saved that effort.

DESIGN. Same model (gemma-4-31B), same LEARNED RULES per game (held constant -- we are testing
perception, not rule-learning), three perception conditions:
  A (naive)   : only the HUD/decoy the segmentation actually surfaced (the observed failure input).
  B (+entities): the naive view PLUS the true entities, but the HUD is NOT labeled as a counter.
  C (oracle)  : the true entities PLUS the HUD explicitly identified as a non-goal counter.
For each we ask gemma for its single best GOAL hypothesis and judge (offline) whether it names the
true win condition. Crucially the perception facts NEVER contain the goal itself (no "reach the exit")
-- only entity presence/role, exactly what a detector would deliver. The model must still INFER the
goal from correct perception. Public-game source facts were read for OFFLINE DEV ANALYSIS only
(authorized per CLAUDE.md public-games source-reading discipline); never used in the hidden submission.

inference_substrate: live_llm_inference (loads + runs gemma-4-31B GGUF on GPU).
verifier_is_oracle: False (no verifier scores anything here; this is a perception counterfactual).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

# Per-game specs, sourced from the REQ-ARC-WMTE-5831 source-grounded analyst reports.
#   rules      : the RULES the model genuinely learned in the v5 sweep (held constant across conditions).
#   naive      : what the segmentation actually surfaced (the HUD/decoy) -- condition A perception.
#   entities   : the true game entities (presence + role), what a perfect detector would output. NO goal.
#   hud_label  : the same entities PLUS the HUD explicitly identified as a non-goal counter -- condition C.
#   true_win   : the true win condition (for offline judging only; never shown to the model).
#   perception_blocked: True if the diagnosis root cause was perception (vs dynamics-not-learned).
GAMES: dict[str, dict] = {
    "sc25": {
        "rules": "Action 6 at (30,55) toggles values between 2 and 14 in a block near (54,29).",
        "naive": "Value cells at columns 62-63 change as you act; a small 3x3 value grid toggles 2<->14.",
        "entities": "There is a movable PLAYER avatar (a small facing sprite) that moves with actions 1/2/3/4. There is a distinct EXIT tile (a bordered door) elsewhere on the board. The value cells at columns 62-63 form a meter that fills as you act. The 3x3 value grid is a spell tool.",
        "hud_label": "There is a movable PLAYER avatar (a small facing sprite) that moves with actions 1/2/3/4. There is a distinct EXIT tile (a bordered door) elsewhere on the board. The value cells at columns 62-63 are a MANA/MOVE METER that simply counts your actions -- they are a status readout, not part of the puzzle. The 3x3 value grid is a spell tool.",
        "true_win": "navigate the player avatar onto the exit tile",
        "perception_blocked": True,
    },
    "tu93": {
        "rules": "Actions 1 and 3 consume cells with value 6.",
        "naive": "Cells with value 6 (in row 63) are consumed as you act.",
        "entities": "There is a movable PLAYER token (color 9) that moves with actions 1/2/3/4. There is a distinct EXIT block (color 14). There are rail cells (color 2). The value-6 cells in row 63 deplete as you act.",
        "hud_label": "There is a movable PLAYER token (color 9) that moves with actions 1/2/3/4. There is a distinct EXIT block (color 14). There are rail cells (color 2). The value-6 cells in row 63 are a STEP-COUNTER that depletes as you act -- a status readout, not part of the puzzle.",
        "true_win": "navigate the player token onto the exit block",
        "perception_blocked": True,
    },
    "ls20": {
        "rules": "Action 1 recolors floor 3/5 to 12; actions 1 and 2 are inverses; some actions swap 3<->11 or 11->3.",
        "naive": "Floor cells change color (3/5 -> 12) as you move around.",
        "entities": "There is a movable PLAYER avatar (color 9). When it steps onto special gadget tiles its shape, color, and rotation change. There are distinct GOAL-PAD tiles elsewhere. You see a 16x16 window over a larger maze.",
        "hud_label": "There is a movable PLAYER avatar (color 9). When it steps onto special gadget tiles its shape, color, and rotation change. There are distinct GOAL-PAD tiles, each requiring a specific shape/color/rotation. The floor recoloring you see is just your avatar moving over floor cells -- not the objective. You see a 16x16 window over a larger maze.",
        "true_win": "morph the avatar's shape/color/rotation to match a goal pad, then stand on it",
        "perception_blocked": True,
    },
    "cn04": {
        "rules": "Action 6 at trigger coordinates toggles/clears values in other cells.",
        "naive": "Cells of colors 4, 12, and 14 change when you click certain trigger coordinates.",
        "entities": "There are several distinct PIECE objects, some carrying terminal MARKER cells (colors 8 and 13). Pieces can be MOVED or ROTATED to align. Colors 4/12/14 are body/background fill. Action-6 selection re-renders a piece.",
        "hud_label": "There are several distinct PIECE objects, some carrying terminal MARKER cells (colors 8 and 13). Pieces can be MOVED or ROTATED to align. Colors 4/12/14 are just body/background fill (decoys) -- changing them is not the objective. Action-6 selection merely re-renders a piece; it does not 'clear' the board.",
        "true_win": "move/rotate pieces so their terminal markers pair up (spatial endpoint pairing)",
        "perception_blocked": True,
    },
    "lf52": {
        "rules": "Actions 1 and 6 at various coordinates change cells in row 0 from 0 to 1.",
        "naive": "Row-0 cells turn to state 1 as you act.",
        "entities": "There are several distinct colored BLOCK objects on the board. Clicking (action 6) ON a block selects it and reveals aim-lines toward an adjacent matching block; clicking an aim-line jumps the block into the matching one, removing one block. Row 0 fills as you act.",
        "hud_label": "There are several distinct colored BLOCK objects on the board. Clicking (action 6) ON a block selects it and reveals aim-lines toward an adjacent matching block; clicking an aim-line jumps the block into the matching one, removing one block. Row 0 is a MOVE-COUNTER bar that fills as you act -- a status readout, not the game board.",
        "true_win": "merge blocks until only one (goal count) remains",
        "perception_blocked": True,
    },
    "bp35": {
        "rules": "Action 6 at (x, y) changes cell (63, x) to color 15.",
        "naive": "Row 63 fills with color-15 cells as you click.",
        "entities": "There is a single PLAYER sprite: action 3 steps it left, action 4 steps it right, and after each step it FALLS under gravity until it lands on a solid block. There is a distinct GEM tile. Spike tiles are deadly. Row 63 fills with color-15 as you act.",
        "hud_label": "There is a single PLAYER sprite: action 3 steps it left, action 4 steps it right, and after each step it FALLS under gravity until it lands on a solid block. There is a distinct GEM tile. Spike tiles are deadly. Row 63 is a MOVE-BUDGET bar that fills as you act; reaching 64 moves ENDS the attempt (it is a lose-timer, not the objective).",
        "true_win": "maneuver the player sprite (via gravity) onto the gem tile",
        "perception_blocked": False,  # dynamics (gravity) -- contrast case
    },
    "r11l": {
        "rules": "Action 6 at (27,57) toggles cells (21,0) through (42,0) between 0 and 5.",
        "naive": "A band of cells (21,0)-(42,0) toggles between 0 and 5 when you click (27,57).",
        "entities": "There are several distinct PIECE objects and matching TARGET-PAD tiles. Interaction is a TWO-click: the first click SELECTS a piece, the second click MOVES it. A single click only recolors the selection band.",
        "hud_label": "There are several distinct PIECE objects and matching TARGET-PAD tiles. Interaction is a TWO-click: the first click SELECTS a piece, the second click MOVES it. A single click only recolors the selection band -- that recolor is not the objective.",
        "true_win": "slide each piece onto its matching target pad",
        "perception_blocked": False,  # dynamics (two-click select-then-move) -- contrast case
    },
    "ft09": {
        "rules": "Action 6 appears to do nothing.",
        "naive": "Action 6 appears to do nothing.",
        "entities": "There are small colored BLOCK objects. Action-6 clicking has an effect ONLY when it lands exactly on a colored block; each such click makes a small local color change. There is a hidden per-tile pattern constraint over the blocks.",
        "hud_label": "There are small colored BLOCK objects. Action-6 clicking has an effect ONLY when it lands exactly on a colored block; each such click makes a small local color change. The win depends on a hidden adjacency-color pattern over the blocks (a constraint you must satisfy), not on any counter.",
        "true_win": "satisfy the hidden adjacency-color pattern constraint over the tiles",
        "perception_blocked": False,  # deep hypothesis-coverage wall -- contrast case
    },
}

GOAL_PROMPT = (
    "You are figuring out how to WIN an unknown grid game (level up / advance). "
    "Actions: 1=up 2=down 3=left 4=right 6=click.\n"
    "You have learned these RULES by playing:\n{rules}\n\n"
    "Here is what you PERCEIVE on the board right now:\n{perception}\n\n"
    "Based ONLY on the rules and what you perceive, state your single BEST hypothesis for the GOAL -- "
    "the specific thing you must achieve to complete the level and advance. One concise line, concrete. "
    "Do not restate the rules.\nGOAL:"
)


def _goal(proposer, rules: str, perception: str, seed: int) -> str:
    from carnot.agentic.arc_greedy_direct_agent import _complete

    prompt = GOAL_PROMPT.format(rules=rules, perception=perception)
    ok, text = _complete(proposer, prompt, max_tokens=60, stop=["\n"], seed=seed)
    return text.strip() if ok else f"(completion failed: {text[:80]})"


def main() -> int:
    port = int(os.environ.get("ABLATION_PORT", "8957"))
    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")  # outer-loop owns GPU 1
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    # PRECONDITION is handled by LocalGGUFProposer._ensure_server(): if the GGUF is uncached or the
    # llama-server fails to start, _complete() returns (False, reason) -- an honest failure datum, never
    # a fabricated goal. So no separate cache check is needed.
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
    seeds = [5831, 5832]
    per_game = []
    for game, spec in GAMES.items():
        conds = {
            "A_naive": spec["naive"],
            "B_entities": spec["entities"] + "\nObserved side-effect while acting: " + spec["naive"],
            "C_oracle": spec["hud_label"],
        }
        goals: dict[str, list[str]] = {}
        for cond, perception in conds.items():
            goals[cond] = [_goal(proposer, spec["rules"], perception, s) for s in seeds]
            for g in goals[cond]:
                print(f"[{game}] {cond}: {g}")
        per_game.append(
            {
                "game": game,
                "perception_blocked": spec["perception_blocked"],
                "true_win": spec["true_win"],
                "goals": goals,
            }
        )
        print()

    art = {
        "experiment": "outer_loop_arc_oracle_perception_goal_ablation",
        "experiment_id": "REQ-ARC-WMTE-5832",
        "run_date": "2026-07-23",
        "schema": "carnot.arc_oracle_perception_goal_ablation.v1",
        "title": "Does handing gemma-4-31B the correct entities (oracle perception) flip its GOAL hypothesis to the true win condition? 3-condition ablation over 8 games.",
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "random_seed": 5831,
        "random_seeds_used": seeds,
        "model_specs": [
            {"name": "gemma-4-31B-it-GGUF", "repo": "unsloth/gemma-4-31B-it-GGUF", "gpu": 1, "port": port}
        ],
        "methodology_note": "Same model + same learned RULES per game (held constant); only the PERCEPTION varies (A naive HUD/decoy, B +true entities, C +entities+HUD-labeled). Perception facts contain entity presence/role ONLY, never the goal. Source facts read for offline dev analysis (authorized), never used in the hidden submission. Judging of goal-match is done offline by the outer loop.",
        "conditions": {
            "A_naive": "only the HUD/decoy the segmentation actually surfaced",
            "B_entities": "naive view + true entities (HUD not labeled)",
            "C_oracle": "true entities + HUD explicitly identified as a non-goal counter",
        },
        "per_game": per_game,
        "duration_s": round(time.time() - t0, 2),
    }
    import hashlib

    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(art, sort_keys=True).encode()
    ).hexdigest()
    out = ROOT / "results" / "outer_loop_arc_oracle_perception_goal_ablation_20260723.json"
    out.write_text(json.dumps(art, indent=2))
    print("wrote", out, f"({art['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
