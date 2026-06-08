"""ARC-AGI-3 policy: local MULTIMODAL Gemma-4 (sees the rendered frame) — the sovereign,
fast, vision-grounded generator (operator direction 2026-06-08).

Why this over codex-text (retired): codex exec is text-only (reasons over a digit-string,
clicks blindly), ~12-120s/call, closed + shares conductor quota. Gemma-4 E2B-it is MULTIMODAL
(Gemma4ForConditionalGeneration, vision_config), LOCAL/sovereign, and ~0.6s/inference — it
SEES the frame. Verifier layers on top as the efficiency/routing layer (future).

Renders the 64x64 ARC color grid -> RGB image (ARC palette, upscaled), feeds image + game
context (available actions, levels, recent action->outcome history) to Gemma-4, parses the
action. Model loaded ONCE (module-cached), reused across all steps/games.

  .venv/bin/python scripts/experiments/arc3_gemma_policy.py --model E2B --n_games 5 --max_actions 60
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
MODELS = {"E2B": "google/gemma-4-E2B-it", "E4B": "google/gemma-4-E4B-it",
          "12B": "unsloth/gemma-4-12B-it-GGUF"}  # 12B is GGUF (llama.cpp path, TODO)
ARC_PAL = [(0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0),
           (170, 170, 170), (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37),
           (255, 255, 255), (100, 60, 40), (200, 120, 200), (90, 140, 90), (140, 90, 40), (60, 60, 90)]

_MODEL = {"proc": None, "model": None, "id": None}


def _render(frame, scale=10):
    import numpy as np
    from PIL import Image
    arr = np.array(frame.frame)
    if arr.ndim == 3:
        arr = arr[-1]
    h, w = arr.shape
    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            img[i * scale:(i + 1) * scale, j * scale:(j + 1) * scale] = ARC_PAL[int(arr[i, j]) % 16]
    return Image.fromarray(img)


def _load(model_key):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    mid = MODELS[model_key]
    if _MODEL["id"] != mid:
        t0 = time.time()
        _MODEL["proc"] = AutoProcessor.from_pretrained(mid)
        _MODEL["model"] = AutoModelForImageTextToText.from_pretrained(
            mid, dtype=torch.bfloat16, device_map="cuda").eval()
        _MODEL["id"] = mid
        print(f"[gemma] loaded {mid} in {time.time()-t0:.0f}s", flush=True)
    return _MODEL["proc"], _MODEL["model"]


def gemma_action(frame, ctx, model_key="E2B", max_new=40):
    """Multimodal policy: render frame -> Gemma-4 sees it + context -> parse ACTION."""
    import torch
    from arcengine.enums import GameAction
    proc, model = _load(model_key)
    by_id = {a.value: a for a in GameAction}
    av = list(getattr(frame, "available_actions", []) or [])
    lv = int(getattr(frame, "levels_completed", 0) or 0)
    hist = ctx.setdefault("mem", {"history": []})["history"]
    img = _render(frame)
    click = "ACTION6 is a CLICK needing x,y in 0-63 (image is the 64x64 grid upscaled 10x). " if 6 in av else ""
    txt = (
        "You are playing an ARC-AGI-3 interactive grid puzzle. The image is the CURRENT frame. "
        "Solve levels by choosing actions; infer the rule from how the grid changes after each action. "
        f"Levels solved so far: {lv}. Available actions: {av}. {click}"
        f"Recent action->outcome: {' | '.join(hist[-6:]) if hist else '(none yet)'}. "
        "Look at the frame and choose the next action to make progress. "
        "Reply with EXACTLY one line: 'ACTION:<n>' or 'ACTION:6 x=<x> y=<y>'."
    )
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    resp = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    m = None
    for line in resp.splitlines():
        mm = re.search(r"ACTION:\s*([0-7])(?:\s+x=\s*(\d+)\s+y=\s*(\d+))?", line, re.I)
        if mm:
            m = mm
    if not m:
        return None, None, resp[:60]
    a = int(m.group(1))
    data = {"x": int(m.group(2)), "y": int(m.group(3))} if a == 6 and m.group(2) else None
    return by_id.get(a, GameAction.ACTION1), data, resp[:60]


def run(model_key="E2B", n_games=5, max_actions=60, seed=0, write=True):
    import random
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameState
    started = time.time()
    rng = random.Random(seed)
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    info = {getattr(e, "game_id", None): (getattr(e, "baseline_actions", None) or []) for e in arc.get_environments()}
    games = list(info)[:n_games]
    per_game, lat = [], []
    for g in games:
        base = info[g]
        env = arc.make(g)
        f = env.reset()
        ctx = {"mem": {"history": []}}
        levels = int(getattr(f, "levels_completed", 0) or 0)
        actions = 0
        for step in range(max_actions):
            st = getattr(f, "state", None)
            if st in (GameState.WIN, GameState.GAME_OVER):
                break
            t0 = time.time()
            action, data, raw = gemma_action(f, ctx, model_key=model_key)
            lat.append(time.time() - t0)
            if action is None:
                break
            before = levels
            f = env.step(action, data=data)
            actions += 1
            after = int(getattr(f, "levels_completed", 0) or 0) if f is not None else before
            outcome = ("LEVEL_UP" if after > before else
                       "game_over" if getattr(f, "state", None) == GameState.GAME_OVER else "no_change")
            ctx["mem"]["history"].append(
                f"ACTION:{action.value}{(' x='+str(data['x'])+' y='+str(data['y'])) if data else ''}->{outcome}")
            levels = after
            if getattr(f, "state", None) == GameState.GAME_OVER:
                break
        r = {"game_id": g, "levels_solved": levels, "win_levels": len(base),
             "actions_used": actions, "final_state": str(getattr(f, "state", "?"))}
        per_game.append(r)
        print(f"  {g:18s} levels={levels}/{len(base)} actions={actions}/{max_actions} state={r['final_state']}", flush=True)

    total = sum(r["levels_solved"] for r in per_game)
    total_win = sum(r["win_levels"] for r in per_game)
    art = {
        "experiment": "arc3_gemma_policy", "title": f"arc3_gemma_{model_key}_multimodal_policy",
        "honest_verdict": (f"complete: gemma_{model_key}_multimodal_levels{total}of{total_win}"
                           f"_beats_floor={total > 0}"),
        "inference_substrate": "offline_arc_agi3_plus_local_gemma4_multimodal_policy",
        "model": MODELS[model_key], "policy": f"gemma4_{model_key}_multimodal",
        "n_games": len(games), "max_actions": max_actions,
        "ACCURACY_total_levels_solved": total, "ACCURACY_total_win_levels": total_win,
        "mean_inference_latency_s": round(sum(lat) / len(lat), 3) if lat else None,
        "n_inferences": len(lat), "per_game": per_game,
        "submitted_to_leaderboard": False, "duration_s": round(time.time() - started, 1),
        "note": ("Local multimodal policy (sees the frame). vs floor random/object_click=0/183. "
                 "Verifier-as-efficiency-layer layers on next. Quota-gate: offline number; online "
                 "only when it beats TRM baseline + best prior Carnot submission."),
    }
    if write:
        (REPO / "results" / f"arc3_gemma_policy_{model_key}.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\n-> {art['honest_verdict']} | mean_latency={art['mean_inference_latency_s']}s "
          f"over {art['n_inferences']} inferences")
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(MODELS), default="E2B")
    ap.add_argument("--n_games", type=int, default=5)
    ap.add_argument("--max_actions", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(model_key=a.model, n_games=a.n_games, max_actions=a.max_actions, seed=a.seed)
