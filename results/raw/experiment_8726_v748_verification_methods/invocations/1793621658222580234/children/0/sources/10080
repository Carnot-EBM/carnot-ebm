#!/usr/bin/env python3
"""#8 Hidden-state diagnostic — is the hard tail grid-only-determined, or hidden-state-bound?

The sprint showed all three generation levers (#1 heuristic / #2 QD / #3 LLM-goal) null on
wa30/sb26. Leading hypothesis: those games have HIDDEN STATE (GAP-ARCH-GRID-ONLY-STATE) -> no
grid-only generator can crack them. This CHEAP diagnostic measures it directly via DETERMINISM:
does the same OBSERVED STATE + same ACTION ever lead to DIFFERENT next-states? If yes, the
observed grid does not capture the full state -> there is hidden state.

Measured on BOTH the RAW grid (full pixels, incl. any HUD) and the LOGICAL grid (downsampled
game grid, may drop a HUD), so we can tell WHERE the deciding state lives:
  - logical NON-deterministic but raw deterministic -> the deciding state is VISIBLE in the raw
    grid (a HUD/register) but DROPPED by logical perception -> fix = PERCEPTION (read the HUD).
  - BOTH non-deterministic -> TRULY hidden state (not visible even in raw) -> state augmentation.
  - BOTH deterministic -> NO hidden state; the wall is search/goal-bound, not state.

Positive control: ka59 (known StepCounter-HUD game per the registry) SHOULD read
"hidden state visible in raw". Honest, OFFLINE, zero quota, sub-minute. verifier_is_oracle: false.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
from carnot.agentic.arc_graph_explore import rich_action_candidates, _warm
from carnot.agentic.arc_executable_world_model import detect_cell, to_logical

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "experiment_8_hidden_state_diagnostic.json"


def _ok(frame) -> bool:
    try:
        return np.asarray(grid_of(frame)).ndim == 2
    except Exception:
        return False


def _rawh(frame):
    return frame_hash(np.asarray(grid_of(frame)))


def _logh(frame, cell):
    return frame_hash(np.asarray(to_logical(grid_of(frame), cell)))


def collect_transitions(game: str, cell, n_rollouts: int, steps: int, rng) -> list:
    """Several random salient walks (reversible moves naturally revisit logical configs while a HUD
    advances) -> (state, action) pairs RECUR via different histories, the precondition for the test."""
    arc = kit.offline_arcade()
    trans = []
    for _ in range(n_rollouts):
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        f = _warm(env, False)
        for _ in range(steps):
            if not _ok(f):
                break
            cands = rich_action_candidates(f)
            if not cands:
                break
            c = cands[rng.randrange(min(len(cands), 8))]
            rh, lh = _rawh(f), _logh(f, cell)
            akey = (int(c.action_id), json.dumps(c.data, sort_keys=True))
            nf = env.step(_game_action(GameAction, int(c.action_id)), data=c.data)
            if nf is None or not _ok(nf):
                break
            trans.append((rh, lh, akey, _rawh(nf), _logh(nf, cell)))
            f = nf
    return trans


def determinism(trans: list, state_idx: int, next_idx: int) -> dict:
    """Group by (state_hash, action); among groups that RECUR, what fraction map to >1 distinct
    next-state? High -> the observed state+action does NOT determine the outcome -> hidden state."""
    counts = defaultdict(int)
    nexts = defaultdict(set)
    for t in trans:
        counts[(t[state_idx], t[2])] += 1
        nexts[(t[state_idx], t[2])].add(t[next_idx])
    recur = [(k, nexts[k]) for k, c in counts.items() if c >= 2]
    nondet = [k for k, ns in recur if len(ns) >= 2]
    return {
        "n_recurring_state_action_groups": len(recur),
        "n_nondeterministic_groups": len(nondet),
        "nondeterminism_rate": round(len(nondet) / len(recur), 3) if recur else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=str, default="wa30,sb26,ka59")
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()
    t0 = time.time()
    games = [g.strip() for g in args.games.split(",") if g.strip()]

    rows = []
    for game in games:
        rng = random.Random(args.seed + hash(game) % 9999)
        t1 = time.time()
        arc = kit.offline_arcade()
        f0 = _warm(arc.make(game, scorecard_id=arc.open_scorecard()), False)
        if not _ok(f0):
            rows.append({"game": game, "diagnosis": "blocked_degenerate_start"}); continue
        cell = detect_cell(grid_of(f0))
        trans = collect_transitions(game, cell, args.rollouts, args.steps, rng)
        raw = determinism(trans, 0, 3)     # raw grid hash -> next raw
        log = determinism(trans, 1, 4)     # logical grid hash -> next logical
        rr, lr = raw["nondeterminism_rate"], log["nondeterminism_rate"]
        # diagnosis (require enough recurrence to be confident)
        enough = (log["n_recurring_state_action_groups"] >= 5)
        if not enough:
            diagnosis = "inconclusive_too_few_recurring_state_action_pairs"
        elif lr is not None and lr >= 0.2 and (rr is None or rr < 0.1):
            diagnosis = "hidden_state_VISIBLE_in_raw_grid_perception_dropped_fix_perception"
        elif lr is not None and lr >= 0.2 and rr is not None and rr >= 0.2:
            diagnosis = "TRULY_hidden_state_needs_state_augmentation"
        elif lr is not None and lr < 0.1:
            diagnosis = "grid_deterministic_NO_hidden_state_wall_is_search_or_goal_bound"
        else:
            diagnosis = "ambiguous"
        row = {
            "game": game, "n_transitions": len(trans),
            "raw_grid": raw, "logical_grid": log,
            "raw_nondeterminism_rate": rr, "logical_nondeterminism_rate": lr,
            "diagnosis": diagnosis, "secs": round(time.time() - t1, 1),
        }
        rows.append(row)
        print(f"  [{game}] logical_nondet={lr} (recur={log['n_recurring_state_action_groups']}) "
              f"raw_nondet={rr} (recur={raw['n_recurring_state_action_groups']}) "
              f"-> {diagnosis} [{row['secs']}s]", flush=True)

    # ka59 positive control check
    ka = next((r for r in rows if r["game"] == "ka59"), None)
    control_ok = ka is not None and "hidden_state_VISIBLE_in_raw" in ka.get("diagnosis", "")
    artifact = {
        "experiment": "experiment_8_hidden_state_diagnostic",
        "honest_verdict": "complete: hidden_state_determinism_diagnostic_ran",
        "verifier_is_oracle": False,
        "inference_substrate": "offline_arc_search",
        "random_seed": args.seed,
        "games": games,
        "positive_control_ka59_detected_hud_hidden_state": control_ok,
        "rows": rows,
        "duration_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\n  ka59 positive-control (should detect HUD hidden state): {control_ok}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
