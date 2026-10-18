"""Feasibility test: codex (gpt-5.5) bridge as the ARC-AGI-3 rule-inducing POLICY (option B).

Plays ONE game OFFLINE with codex deciding each action, BOUNDED tight (codex exec is ~10s/call
and shares the conductor's quota, so per-step over a full 25-game eval is impractical — this is a
feasibility probe: can a frontier LLM, given the frame + history, INDUCE the mechanic and make
progress past the 0/183 random+pruner floor?).

Bridge = `codex exec --color never --model gpt-5.5 --dangerously-bypass-approvals-and-sandbox`
(same invocation the conductor uses). Frame serialized compactly (active-region grid + objects +
available actions + recent action->outcome history). Verifier role (action-pruning) folded in by
offering codex the OBJECT click-targets, not raw 64x64.

  .venv/bin/python scripts/experiments/arc3_codex_policy_test.py --game vc33-5430563c --max_actions 15
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
CODEX = ["codex", "exec", "--color", "never", "--model", "gpt-5.5",
         "--dangerously-bypass-approvals-and-sandbox", "--cd", "/tmp", "--ephemeral"]


def _grid_and_objects(frame):
    import numpy as np
    arr = np.array(frame.frame)
    if arr.ndim == 3:
        arr = arr[-1]
    vals, counts = np.unique(arr, return_counts=True)
    bg = int(vals[counts.argmax()])
    ys, xs = np.where(arr != bg)
    if len(ys) == 0:
        return arr, bg, []
    # crop to active bounding box to keep the prompt small
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    crop = arr[y0:y1 + 1, x0:x1 + 1]
    # objects (connected non-bg components) as click candidates, in full-grid coords
    from arc3_offline_eval import _objects  # reuse the perception
    objs = _objects(frame)
    return crop, bg, [(int(y), int(x)) for (y, x) in objs], (int(y0), int(x0))


def _serialize(frame, baseline, history):
    crop, bg, objs, origin = _grid_and_objects(frame)
    rows = "\n".join("".join(str(int(c)) for c in r) for r in crop.tolist())
    av = list(getattr(frame, "available_actions", []) or [])
    lv = int(getattr(frame, "levels_completed", 0) or 0)
    obj_lines = "; ".join(f"obj@(y={y},x={x})" for (y, x) in objs[:20])
    hist = " | ".join(history[-6:]) if history else "(none yet)"
    return (
        f"ARC-AGI-3 interactive grid game. Goal: solve levels by choosing actions; figure out the "
        f"rule from how the grid changes. Background color={bg}. Levels solved so far={lv}.\n"
        f"Active grid region (cropped, origin y={origin[0]} x={origin[1]}; digits=colors):\n{rows}\n"
        f"Object click-targets (full-grid coords): {obj_lines or '(none)'}\n"
        f"Available actions={av}. ACTION6 is a CLICK needing x,y (0-63). Other actions take no data.\n"
        f"Recent action->outcome history: {hist}\n"
        f"Reply with EXACTLY one line and nothing else: 'ACTION:<n>' or 'ACTION:6 x=<x> y=<y>'."
    )


def _ask_codex(prompt, timeout=120):
    t0 = time.time()
    try:
        r = subprocess.run(CODEX, input=prompt, capture_output=True, text=True, timeout=timeout)
        out = (r.stdout or "")
    except (subprocess.TimeoutExpired, OSError) as e:
        return None, None, round(time.time() - t0, 1), f"codex_error:{type(e).__name__}"
    # parse the LAST ACTION:... line codex emits
    m = None
    for line in out.splitlines():
        mm = re.search(r"ACTION:\s*([0-7])(?:\s+x=\s*(\d+)\s+y=\s*(\d+))?", line, re.I)
        if mm:
            m = mm
    if not m:
        return None, None, round(time.time() - t0, 1), "no_action_parsed"
    a = int(m.group(1))
    data = ({"x": int(m.group(2)), "y": int(m.group(3))} if a == 6 and m.group(2) else None)
    return a, data, round(time.time() - t0, 1), "ok"


def run(game="vc33-5430563c", max_actions=15, write=True):
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction, GameState
    import sys
    sys.path.insert(0, str(REPO / "scripts" / "experiments"))
    started = time.time()
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    info = {getattr(e, "game_id", None): (getattr(e, "baseline_actions", None) or []) for e in arc.get_environments()}
    baseline = info.get(game, [])
    by_id = {a.value: a for a in GameAction}
    env = arc.make(game)
    f = env.reset()
    levels = int(getattr(f, "levels_completed", 0) or 0)
    history, trace = [], []
    print(f"[codex-policy] game={game} baseL0={baseline[0] if baseline else '?'} max_actions={max_actions}", flush=True)
    for step in range(max_actions):
        st = getattr(f, "state", None)
        if st in (GameState.WIN, GameState.GAME_OVER):
            print(f"  terminal: {st}"); break
        prompt = _serialize(f, baseline, history)
        a, data, lat, status = _ask_codex(prompt)
        if a is None:
            # RESILIENT: a codex timeout/parse-miss falls back to the object-click pruner
            # (codex+verifier hybrid) so one slow call doesn't end the game.
            from arc3_offline_eval import object_click_policy
            import random as _r
            ctx = {"grid_h": __import__("numpy").array(f.frame)[-1].shape[0] if hasattr(f, "frame") else 64,
                   "grid_w": __import__("numpy").array(f.frame)[-1].shape[1] if hasattr(f, "frame") else 64,
                   "mem": {"obj_i": step, "kb_i": step}}
            fb_action, data = object_click_policy(f, ctx, _r.Random(step))
            a = fb_action.value if fb_action is not None else 1
            status = f"codex_{status}->objclick_fallback"
            print(f"  step {step}: codex {status} ({lat}s) -> fallback action {a}", flush=True)
        before = levels
        f = env.step(by_id.get(a, GameAction.ACTION1), data=data)
        after = int(getattr(f, "levels_completed", 0) or 0) if f is not None else before
        changed = "LEVEL_UP" if after > before else ("game_over" if getattr(f, "state", None) == GameState.GAME_OVER else "no_level_change")
        levels = after
        hist_entry = f"ACTION:{a}{(' x='+str(data['x'])+' y='+str(data['y'])) if data else ''}->{changed}"
        history.append(hist_entry)
        trace.append({"step": step, "action": a, "data": data, "latency_s": lat, "outcome": changed, "levels": levels})
        print(f"  step {step}: {hist_entry} (codex {lat}s) levels={levels}", flush=True)
        if getattr(f, "state", None) == GameState.GAME_OVER:
            print("  GAME_OVER"); break

    solved = levels > 0
    art = {
        "experiment": "arc3_codex_policy_test", "game": game,
        "honest_verdict": (f"complete: codex_policy_{'SOLVED_'+str(levels)+'_levels' if solved else 'no_solve'}"
                           f"_beats_floor={solved}"),
        "inference_substrate": "offline_arc_agi3_plus_codex_gpt55_bridge_policy",
        "policy": "codex_gpt5.5", "levels_solved": levels, "max_actions": max_actions,
        "baseline_actions": baseline, "trace": trace,
        "mean_codex_latency_s": round(sum(t["latency_s"] for t in trace) / len(trace), 1) if trace else None,
        "n_codex_calls": len(trace), "submitted_to_leaderboard": False,
        "duration_s": round(time.time() - started, 1),
        "note": ("Feasibility probe. codex per-step ~10s + shares conductor quota -> NOT for full "
                 "25-game per-step play. If it SOLVES here, the frontier-LLM reference is real and "
                 "we then move the policy to a cheap local model + verifier routing (the thesis)."),
    }
    if write:
        (REPO / "results" / f"arc3_codex_policy_{game.split('-')[0]}.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\n-> {art['honest_verdict']} (levels={levels}, {len(trace)} codex calls, "
          f"mean {art['mean_codex_latency_s']}s/call)")
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="vc33-5430563c")
    ap.add_argument("--max_actions", type=int, default=15)
    raise SystemExit(0 if run(**vars(ap.parse_args())) else 1)
