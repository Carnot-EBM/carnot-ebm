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
          "12B": "google/gemma-4-12B-it"}  # dense 12B (transformers, multimodal); 4-bit to fit 24GB
_QUANT_4BIT = {"12B"}  # models too big for bf16 on a 24GB GPU -> load 4-bit
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
        kw = dict(device_map="cuda")
        if model_key in _QUANT_4BIT:  # 4-bit to fit a 12B on 24GB
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                # do NOT quantize the vision tower/projector -> avoids the Byte/layernorm crash
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "vision_model"])
            kw["dtype"] = torch.bfloat16
        else:
            kw["dtype"] = torch.bfloat16
        _MODEL["model"] = AutoModelForImageTextToText.from_pretrained(mid, **kw).eval()
        _MODEL["id"] = mid
        print(f"[gemma] loaded {mid} ({'4bit' if model_key in _QUANT_4BIT else 'bf16'}) "
              f"in {time.time()-t0:.0f}s", flush=True)
    return _MODEL["proc"], _MODEL["model"]


def gemma_action(frame, ctx, model_key="E2B", reasoning=False):
    """Multimodal policy: render frame -> Gemma-4 sees it + context -> ACTION.
    reasoning=True: chain-of-thought + a carried-forward NOTES hypothesis (the rule-induction loop)."""
    import torch
    from arcengine.enums import GameAction
    proc, model = _load(model_key)
    by_id = {a.value: a for a in GameAction}
    av = list(getattr(frame, "available_actions", []) or [])
    lv = int(getattr(frame, "levels_completed", 0) or 0)
    mem = ctx.setdefault("mem", {})
    mem.setdefault("history", [])
    mem.setdefault("notes", "")
    hist = mem["history"]
    img = _render(frame)
    click = "ACTION6 is a CLICK needing x,y in 0-63 (image is the 64x64 grid upscaled 10x). " if 6 in av else ""
    if reasoning:
        txt = (
            "You are SOLVING an ARC-AGI-3 interactive grid puzzle. The image is the CURRENT frame.\n"
            f"Levels solved so far: {lv}. Available actions: {av}. {click}\n"
            f"YOUR RUNNING NOTES on this game's rule (carry/refine): {mem['notes'] or '(none yet — discover it)'}\n"
            f"Recent action->outcome: {' | '.join(hist[-8:]) if hist else '(none yet)'}\n"
            "Think step by step (briefly): (1) what changed after the last action? (2) what is the game's "
            "RULE/GOAL? (3) what single action best makes progress now? Early on, try DIFFERENT actions to "
            "learn what each does; avoid actions that caused game_over.\n"
            "Output EXACTLY two lines:\nNOTES: <your updated one-sentence rule hypothesis>\n"
            "ACTION:<n>   (or 'ACTION:6 x=<x> y=<y>')"
        )
        max_new = 256
    else:
        txt = (
            "You are playing an ARC-AGI-3 interactive grid puzzle. The image is the CURRENT frame. "
            "Solve levels by choosing actions; infer the rule from how the grid changes. "
            f"Levels solved so far: {lv}. Available actions: {av}. {click}"
            f"Recent action->outcome: {' | '.join(hist[-6:]) if hist else '(none yet)'}. "
            "Reply with EXACTLY one line: 'ACTION:<n>' or 'ACTION:6 x=<x> y=<y>'."
        )
        max_new = 40
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": txt}]}]
    inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                      return_dict=True, return_tensors="pt").to("cuda")
    # some Gemma-4 variants (12B/gemma4_unified) hand pixel_values through as uint8 (Byte) ->
    # vision-tower layernorm crashes; cast to the compute dtype.
    if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.uint8:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    resp = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    if reasoning:  # carry the refined hypothesis forward
        nm = re.search(r"NOTES:\s*(.+)", resp, re.I)
        if nm:
            mem["notes"] = nm.group(1).strip()[:200]
    m = None
    for line in resp.splitlines():
        mm = re.search(r"ACTION:\s*([0-7])(?:\s+x=\s*(\d+)\s+y=\s*(\d+))?", line, re.I)
        if mm:
            m = mm
    if not m:
        return None, None, resp[:80]
    a = int(m.group(1))
    data = {"x": int(m.group(2)), "y": int(m.group(3))} if a == 6 and m.group(2) else None
    return by_id.get(a, GameAction.ACTION1), data, resp[:80]


def run(model_key="E2B", n_games=5, max_actions=60, seed=0, reasoning=False, write=True):
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
            action, data, raw = gemma_action(f, ctx, model_key=model_key, reasoning=reasoning)
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
    mode = "reasoning" if reasoning else "direct"
    art = {
        "experiment": "arc3_gemma_policy", "title": f"arc3_gemma_{model_key}_{mode}_policy",
        "honest_verdict": (f"complete: gemma_{model_key}_{mode}_multimodal_levels{total}of{total_win}"
                           f"_beats_floor={total > 0}"),
        "inference_substrate": "offline_arc_agi3_plus_local_gemma4_multimodal_policy",
        "model": MODELS[model_key], "policy": f"gemma4_{model_key}_multimodal_{mode}",
        "reasoning_mode": reasoning,
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
        (REPO / "results" / f"arc3_gemma_policy_{model_key}_{mode}.json").write_text(
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
    ap.add_argument("--reasoning", action="store_true", help="chain-of-thought + carried NOTES hypothesis")
    a = ap.parse_args()
    run(model_key=a.model, n_games=a.n_games, max_actions=a.max_actions, seed=a.seed, reasoning=a.reasoning)
