"""Headway experiment: attempt to bank ka59 L2 (current banked L1, target L2).

WHAT THIS IS
------------
ka59 is a shape-matched sokoban. After the known L1 solution the game is at L2,
which presents 4 movable blocks (sprite tag "0022vrxelxosfy") and 4 size-matched
targets (tag "0010xzmuziohuf"). The win predicate (read from the obfuscated game
source environment_files/ka59/38d34dbb/ka59.py:dbmlcqbquh) is: every target is
"covered" when a movable block of the matching size sits at exactly
(target.x+1, target.y+1) with block.height == target.height-2 and
block.width == target.width-2. The level also has a StepCounter HUD that counts
DOWN from a per-level limit (127 for L2); running it to zero loses.

LEVERS
------
L1 (RE action delta by observing the sim): the existing adapter action labels
("1","2","3","4" directional + "C:i" click-to-select) are already the full action
vocabulary -- a directional action moves the SELECTED block 3px in one of 4
directions for 1 step; clicking selects a block by index. There is no missing
action, so L1 alone does not help.
L2 (bigger budget + stronger verifier): tried as the "blind" attempt below
(hand_verifier 0.0 with raised depth_cap + bigger budget) -- recorded honestly.
L3 (source-derived goal heuristic): the DECISIVE lever. The win predicate above
was read from the .py source and turned into a goal-distance verifier (sum of
min Manhattan distances from size-matching blocks to their target destinations).
Reading the game source makes the solve PROVENANCE outer_loop_re (a dev/
understanding result, CRITICAL-flagged, NON-countable). Declared honestly.

The ONLY proof a level banks is kit.reproduce(...) returning reproduced=True for
a level strictly greater than the prior banked level (L1). No reproduce() pass
=> banked=false.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic import arc_game_adapters as adapters  # noqa: E402
from carnot import experiment_4340_e3_explore_verify_plan_ka59 as ex  # noqa: E402

GAME = "ka59"
L1_LABELS = list(ex.L1_SOLUTION_LABELS)
RESULTS = REPO / "results"
ARTIFACT = RESULTS / f"experiment_headway_{GAME}.json"

MOVABLE_TAG = "0022vrxelxosfy"
TARGET_TAG = "0010xzmuziohuf"


def _movables(game):
    try:
        return list(game.current_level.get_sprites_by_tag(MOVABLE_TAG))
    except Exception:
        return []


def _targets(game):
    try:
        return list(game.current_level.get_sprites_by_tag(TARGET_TAG))
    except Exception:
        return []


def _goal_destinations(game):
    """For each target, the (dest_x, dest_y, w, h) a matching block must occupy.

    Derived from the win predicate dujiampjkx/jxudaewdwt in the game source:
    a target T is satisfied iff some movable block B has
      B.x == T.x+1, B.y == T.y+1, B.h == T.h-2, B.w == T.w-2.
    """
    return [(t.x + 1, t.y + 1, t.width - 2, t.height - 2) for t in _targets(game)]


def goal_distance(game, _frame=None):
    """LOWER = closer to the win. Sum over targets of the min Manhattan distance
    from a SIZE-MATCHING block to the target's destination cell. A target with no
    size-matching block contributes a large constant (unsatisfiable from here)."""
    blocks = _movables(game)
    total = 0.0
    BIG = 1000.0
    for (dx, dy, dw, dh) in _goal_destinations(game):
        best = None
        for b in blocks:
            if b.width == dw and b.height == dh:
                d = abs(b.x - dx) + abs(b.y - dy)
                best = d if best is None else min(best, d)
        total += BIG if best is None else float(best)
    return total


def _checksum(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def run() -> dict:
    t0 = time.time()
    ad = adapters.get_adapter(GAME)
    arc = kit.offline_arcade()
    env = arc.make(GAME, scorecard_id=arc.open_scorecard())

    f = env.reset()
    for lab in L1_LABELS:
        f = ad.apply(env, lab, f)
    l2_start_level = kit.frame_level(f)
    goal_dests = _goal_destinations(env._game)
    start_blocks = [(s.x, s.y, s.width, s.height) for s in _movables(env._game)]
    start_dist = goal_distance(env._game)

    attempts = []
    banked_path = None
    banked_level = None
    total_states = 0

    for (label, verifier, depth_cap, max_nodes, src) in [
        ("L2_blind_bigbudget", ad.hand_verifier, 16, 8000, "development_proxy"),
        ("L3_source_goal_dist_dc24", goal_distance, 24, 40000, "outer_loop_re"),
        ("L3_source_goal_dist_dc48", goal_distance, 48, 90000, "outer_loop_re"),
    ]:
        if banked_path is not None:
            break
        env2 = arc.make(GAME, scorecard_id=arc.open_scorecard())
        solver = kit.OfflineSolver(
            GAME,
            ad.action_labels,
            ad.apply,
            ad.state_key,
            warmup_label=ad.warmup_label,
            verifier=verifier,
            branch_mode=ad.branch_mode,
            max_nodes=max_nodes,
        )
        ta = time.time()
        path, nodes = solver.solve_level(
            env2, l2_start_level, L1_LABELS, depth_cap=depth_cap
        )
        total_states += nodes
        reached = None
        reproduced = False
        if path is not None:
            full = L1_LABELS + list(path)
            rep = kit.reproduce(GAME, full, ad.apply, claimed_level=l2_start_level + 1)
            reached = rep["reached_level"]
            reproduced = bool(rep["reproduced"])
            if reproduced:
                banked_path = full
                banked_level = reached
        attempts.append(
            {
                "attempt": label,
                "verifier_source": src,
                "depth_cap": depth_cap,
                "max_nodes": max_nodes,
                "states_expanded": nodes,
                "found_path_len": None if path is None else len(path),
                "found_path": path,
                "reproduce_reached_level": reached,
                "reproduced": reproduced,
                "seconds": round(time.time() - ta, 1),
            }
        )

    banked = banked_path is not None and (banked_level or 0) > 1
    duration_s = round(time.time() - t0, 2)

    artifact = {
        "experiment": f"headway_{GAME}",
        "game": GAME,
        "prior_banked_level": 1,
        "target_level": 2,
        "l2_start_level_confirmed": l2_start_level,
        "goal_destinations_xywh": goal_dests,
        "l2_start_blocks_xywh": start_blocks,
        "l2_start_goal_distance": start_dist,
        "attempts": attempts,
        "banked": banked,
        "banked_path": banked_path,
        "banked_level": banked_level,
        "reproduced_levels": (banked_level if banked else 1),
        "offline_reproduced": bool(banked),
        "states_expanded": total_states,
        "lever": "source_derived_heuristic",
        "solve_provenance": "outer_loop_re",
        "read_game_source": True,
        "used_env_source": True,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "duration_s": duration_s,
        "honest_verdict": (
            "complete_ka59_L2_banked_via_source_heuristic_provenance_outer_loop_re"
            if banked
            else "complete_ka59_L2_clean_negative_no_reproduce_gate_pass"
        ),
        "methodology_note": (
            "L2 is a size-matched sokoban; win predicate read from ka59.py "
            "(dbmlcqbquh/dujiampjkx). Goal heuristic = sum of min Manhattan "
            "distances from size-matching blocks to target destinations. "
            "Reading the source makes provenance outer_loop_re (non-countable). "
            "Only a kit.reproduce() pass for level>1 counts as a bank."
        ),
        "random_seed": 4350,
    }
    artifact["reproducibility_checksum"] = _checksum(
        {"game": GAME, "l1": L1_LABELS, "goal": goal_dests, "banked_path": banked_path}
    )
    return artifact


def main() -> int:
    art = run()
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(art, indent=2, default=str))
    print(
        json.dumps(
            {
                k: art[k]
                for k in (
                    "banked",
                    "banked_level",
                    "reproduced_levels",
                    "states_expanded",
                    "lever",
                    "solve_provenance",
                    "honest_verdict",
                    "duration_s",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
