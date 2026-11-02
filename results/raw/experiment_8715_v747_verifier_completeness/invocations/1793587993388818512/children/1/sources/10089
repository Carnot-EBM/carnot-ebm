#!/usr/bin/env python3
"""Cross-level knowledge retention: measure the post-level-up sink and test what persists.

WHAT THIS MEASURES
==================
On a level-up, `E3AgentPolicy._begin_level_goal_episode` resets the induction state
(`induced=False`, the transition window, the plan). The agent then re-explores and
re-induces from scratch. Two questions, answered from live offline runs:

1. SINK SIZE. How many actions does each level after the first cost, and how much of
   that cost re-derives knowledge the agent already had? Decomposition per level window:
   - repeat (state, action) pairs: re-walking ground the agent already walked
   - transitions exactly predicted by a model fit ONLY on the PRIOR level's window:
     evidence the new level's dynamics were already known.

2. WHAT PERSISTS. For each consecutive level pair (k, k+1):
   - DYNAMICS: fit `InducedNavWorldModel` on window k, score it with
     `WorldModelVerifier` on window k+1. Compare against the same model fit on k+1
     scored on k+1 (the in-corpus ceiling). Also score real archived LLM-induced
     engines (from the h2h runs) on both windows where the game has one.
   - GOAL: does the prior window's induced goal predicate fire on the NEXT level's
     win state, and how often does it false-positive on non-win states?
   - ACTION SEMANTICS: per-action effect-class signatures (which color-to-color
     rewrites an action causes) at k versus k+1.

WHAT THIS IS NOT. Offline, public games, LLM OFF (`CARNOT_ARC_DISABLE_INDUCTION=1`),
submits nothing, flips no flag. Reads `results/**` archives; writes ONLY into its own
new output directory. The policy construction copies
`scripts/arc_per_level_reset_attribution_capture.py` exactly, so the level costs here
must reproduce that artifact's recorded numbers (a built-in harness check).

Usage:
  outer_loop_arc_cross_level_retention_20260817.py --games tu93,vc33 \
      --seeds 20260724,20260725,20260726 --budget 4000 --out results/arc_cross_level_retention_20260817/capture.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))

# LLM OFF, set before the agent module is imported so no proposer is constructed.
os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
# Redirect the per-game engine store so nothing in this process can touch the real one.
_E3_TMP = tempfile.mkdtemp(prefix="arc_xlevel_e3_")
os.environ["CARNOT_ARC_E3_DIR"] = _E3_TMP

SUBSTRATE = "offline_arcade_live_agent_runtime_self_discovery_no_llm"

ENGINE_ARCHIVES = [
    REPO / "results/arc_qwen38_h2h_stopped_20260817/engine_archive",
    REPO / "results/arc_qwen38_h2h_partial_20260817/engine_snapshots",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _grid_hash(grid) -> str:
    import numpy as np

    a = np.asarray(grid)
    return hashlib.sha1(a.tobytes() + str(a.shape).encode()).hexdigest()[:16]


def _effect_signature(t) -> tuple | None:
    """Effect class of one transition: which (from_color, to_color) rewrites happened.

    Position-free on purpose: an avatar step up at (3,4) and one at (9,9) share a
    signature, so 'was this effect class seen on the prior level' is a semantics
    question, not a coordinates question. Returns None for a no-op or a shape change
    (a shape change is its own class, counted separately).
    """
    import numpy as np

    g, n = np.asarray(t.grid), np.asarray(t.next_grid)
    if g.shape != n.shape:
        return ("shape_change", g.shape, n.shape)
    diff = g != n
    count = int(diff.sum())
    if count == 0:
        return None
    pairs = tuple(sorted(set(zip(g[diff].astype(int).tolist(), n[diff].astype(int).tolist()))))
    bucket = count if count <= 8 else "9+"
    return (pairs, bucket)


def _score_engine(engine, is_done, transitions, label: str) -> dict:
    """Score an engine on a transition window with the live verifier. Never raises."""
    from carnot.agentic import arc_executable_world_model as e3

    out: dict = {"label": label, "n_transitions": len(transitions)}
    if not transitions:
        out["skipped"] = "empty_window"
        return out
    try:
        vr = e3.WorldModelVerifier(list(transitions)).score(engine)
        out["accuracy"] = round(float(vr.accuracy), 4)
        out["cell_recall"] = round(float(vr.cell_recall), 4)
        out["change_fidelity"] = round(float(vr.change_fidelity), 4)
    except Exception as exc:  # a crashing engine is a result, not a crash of the capture
        out["error"] = repr(exc)[:160]
        return out
    if is_done is not None:
        import numpy as np

        win_hits = 0
        win_total = 0
        fp = 0
        fp_total = 0
        for t in transitions:
            is_win_row = int(t.level_after) > int(t.level_before)
            try:
                fired = bool(is_done(np.asarray(t.next_grid)))
            except Exception:
                continue
            if is_win_row:
                win_total += 1
                win_hits += int(fired)
            else:
                fp_total += 1
                fp += int(fired)
        out["goal_fired_on_win_states"] = f"{win_hits}/{win_total}"
        out["goal_false_positive_rate"] = round(fp / fp_total, 4) if fp_total else None
    return out


def _fit_nav(transitions):
    from carnot.agentic.arc_nav_world_model import InducedNavWorldModel

    nav = InducedNavWorldModel.fit(list(transitions))
    is_nav = (
        bool(getattr(nav, "displacement", None)) and getattr(nav, "goal_color", None) is not None
    )
    return nav, is_nav


def _archived_engines(game: str) -> list[dict]:
    """Load every archived LLM-induced engine for this game, via the store loader."""
    from carnot.agentic import arc_executable_world_model as e3

    rows = []
    for archive in ENGINE_ARCHIVES:
        if not archive.is_dir():
            continue
        for path in sorted(archive.glob(f"*__{game}__*.py")):
            slot = Path(_E3_TMP) / game
            slot.mkdir(parents=True, exist_ok=True)
            (slot / "world_model.py").write_text(path.read_text())
            try:
                engine, is_done = e3._load_engine_from(Path(_E3_TMP), game)
                rows.append(
                    {"source": str(path.relative_to(REPO)), "engine": engine, "is_done": is_done}
                )
            except Exception as exc:
                rows.append({"source": str(path.relative_to(REPO)), "load_error": repr(exc)[:160]})
    return rows


def run_cell(game: str, seed: int, budget: int) -> dict:
    """One LIVE scored-path cell, LLM off -- byte-for-byte the reset-attribution recipe."""
    import numpy as np

    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    random.seed(seed)
    np.random.seed(seed % (2**32))
    policy = E3AgentPolicy(game, frontier_discipline_seed=seed)
    t0 = time.time()
    r = lb.run_game(game, policy, budget=budget, variant=0, reflect=None)
    wall = time.time() - t0

    transitions = list(getattr(policy, "transitions", []) or [])
    windows: dict[int, list] = defaultdict(list)
    for t in transitions:
        windows[int(t.level_before)].append(t)
    levels = sorted(windows)

    cell: dict = {
        "game": game,
        "seed": seed,
        "budget": budget,
        "wall_s": round(wall, 2),
        "levels_reached": int(r.get("reached") or 0),
        "per_level_actions": r.get("per_level"),
        "actions_total": int(r.get("actions") or 0),
        "n_transitions": len(transitions),
        "level_windows": {},
        "cross_level": [],
    }

    # ---- per-window sink decomposition -------------------------------------------------
    seen_state_action: set = set()
    seen_next: set = set()
    per_window_sigs: dict[int, Counter] = {}
    for lv in levels:
        rows = windows[lv]
        repeats = 0
        noops = 0
        novel_next = 0
        sigs: Counter = Counter()
        for t in rows:
            key = (_grid_hash(t.grid), int(t.action))
            if key in seen_state_action:
                repeats += 1
            seen_state_action.add(key)
            nh = _grid_hash(t.next_grid)
            if nh not in seen_next:
                novel_next += 1
            seen_next.add(nh)
            sig = _effect_signature(t)
            if sig is None:
                noops += 1
            else:
                sigs[(int(t.action), sig)] += 1
        per_window_sigs[lv] = sigs
        cell["level_windows"][str(lv)] = {
            "n": len(rows),
            "repeat_state_action": repeats,
            "repeat_state_action_fraction": round(repeats / len(rows), 4) if rows else None,
            "noop_fraction": round(noops / len(rows), 4) if rows else None,
            "novel_next_state_fraction": round(novel_next / len(rows), 4) if rows else None,
            "n_distinct_effect_classes": len(sigs),
        }

    # ---- cross-level continuity --------------------------------------------------------
    archived = _archived_engines(game)
    for prev_lv, next_lv in zip(levels, levels[1:]):
        prev_rows, next_rows = windows[prev_lv], windows[next_lv]
        pair: dict = {
            "prev_level": prev_lv,
            "next_level": next_lv,
            "n_prev": len(prev_rows),
            "n_next": len(next_rows),
        }

        # effect-class carryover: how much of the NEXT level's observed dynamics was
        # already exhibited (same action, same effect class) on the PREVIOUS level.
        prev_sigs = set(per_window_sigs[prev_lv])
        next_sigs = per_window_sigs[next_lv]
        known = sum(c for s, c in next_sigs.items() if s in prev_sigs)
        total = sum(next_sigs.values())
        pair["effect_class_carryover_weighted"] = round(known / total, 4) if total else None
        pair["effect_classes_next"] = len(next_sigs)
        pair["effect_classes_next_already_seen"] = sum(1 for s in next_sigs if s in prev_sigs)

        # nav-model transfer (the principled dynamics test on nav-family games).
        # Each fit is guarded on its own: one window's fit crash must not hide the other's.
        try:
            nav_prev, is_nav_prev = _fit_nav(prev_rows)
            pair["nav_fit_prev_is_nav"] = is_nav_prev
            if is_nav_prev:
                eng_p, done_p = nav_prev.as_callables()
                pair["nav_prev_on_prev"] = _score_engine(eng_p, done_p, prev_rows, "in_sample")
                pair["nav_prev_on_next"] = _score_engine(eng_p, done_p, next_rows, "transfer")
        except Exception as exc:
            pair["nav_prev_error"] = repr(exc)[:160]
        try:
            nav_next, is_nav_next = _fit_nav(next_rows)
            if is_nav_next:
                eng_n, done_n = nav_next.as_callables()
                pair["nav_next_on_next"] = _score_engine(eng_n, done_n, next_rows, "ceiling")
        except Exception as exc:
            pair["nav_next_error"] = repr(exc)[:160]

        # archived real LLM engines: the highest-fidelity 'would the carried engine
        # have verified on the next level' evidence available without a GPU.
        arch_rows = []
        for a in archived:
            if "engine" not in a:
                arch_rows.append({"source": a["source"], "load_error": a.get("load_error")})
                continue
            arch_rows.append(
                {
                    "source": a["source"],
                    "on_prev": _score_engine(a["engine"], a["is_done"], prev_rows, "prev"),
                    "on_next": _score_engine(a["engine"], a["is_done"], next_rows, "next"),
                }
            )
        if arch_rows:
            pair["archived_engines"] = arch_rows
        cell["cross_level"].append(pair)

    # navigation diagnostics from the harness (whole-run replay overhead)
    cell["navigation_diagnostics"] = r.get("navigation_diagnostics")
    return cell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--seeds", default="20260724,20260725,20260726")
    ap.add_argument("--budget", type=int, default=4000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.time()
    cells = []
    for game in games:
        for seed in seeds:
            print(f"== {game} seed={seed} budget={args.budget}", flush=True)
            try:
                cells.append(run_cell(game, seed, args.budget))
            except Exception as exc:
                cells.append(
                    {"game": game, "seed": seed, "budget": args.budget, "error": repr(exc)[:300]}
                )
            print(
                f"   done in {cells[-1].get('wall_s', '?')}s "
                f"levels={cells[-1].get('levels_reached')}",
                flush=True,
            )

    agent_path = REPO / "python/carnot/agentic/arc_competition_agent.py"
    artifact = {
        "experiment": "outer_loop_arc_cross_level_retention_20260817",
        "title": "Cross-level retention: sink decomposition + what-persists continuity tests",
        "run_date": time.strftime("%Y-%m-%d"),
        "inference_substrate": SUBSTRATE,
        "llm_enabled": False,
        "random_seeds_used": seeds,
        "budget": args.budget,
        "duration_s": round(time.time() - t0, 2),
        "provenance": {
            "agent_sha256": _sha256(agent_path),
            "harness": "scripts/arc_leaderboard_eval.py:run_game",
            "policy_recipe": "arc_per_level_reset_attribution_capture.py run_cell, reproduced",
            "engine_archives": [str(p.relative_to(REPO)) for p in ENGINE_ARCHIVES if p.is_dir()],
        },
        "n_cells": len(cells),
        "cells": cells,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=1, default=str))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
