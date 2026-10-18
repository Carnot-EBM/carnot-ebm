"""Config rule-inducer LAYER B -- the LLM-scene-reader, GENERALIZED across config games.

Plan: docs/research-notes/arc-config-target-induction-scope-2026-06-18.md (Section 4.1). Layer A
(interaction vocabulary) is done; Layer B is the WIN-RULE -- the relation the configuration must
satisfy, which no generic heuristic captures. This tests whether a LOCAL open-weight GGUF
(offline-legal per Decentralization Rule 1, zero quota) can READ a config scene and INDUCE the win-rule.

This generalizes scripts/experiments/arc3_config_layerb_llm_tr87.py (tr87 = the HARDEST rule-class,
glyph-rewrite, which did NOT ground). Operator move #1: test whether the SAME reader + SAME prompt
scaffolding grounds on a SIMPLER rule-class before concluding the offline model is too weak. The ONLY
changed variable vs the tr87 run is the game (rule difficulty) -- the prompt structure, the proposer,
and the propose-then-ground verifier are identical. Default game = ka59 (bottom-strip recolor-config:
a 1-D 'legend/match' class, shorter banked solve, structurally simpler than tr87's 2-D rewrite).

The Carnot propose-then-ground pattern: the LLM PROPOSES the rule as a checkable Python predicate
`is_win(grid)`; the VERIFIER grounds it -- the induced rule must fire True on the banked WIN config and
False on NON-win configs. Every config game with a banked solve is a ground-truth case (we can score the
induced rule against reality).

Honest test: grounded on a simpler game -> the offline reader IS viable, scaffolding is the lever, not
model strength. Not grounded on ANY game -> a stronger reader (or much heavier scaffolding) is needed --
the known decentralization tradeoff. iGPU port 8920; zero quota; tears down its server.

Usage: python scripts/experiments/arc3_config_layerb_llm.py [game]   (default ka59)
Config games WITH a banked solve (verifiable): ka59 (11-action), sc25 (16), tn36 (102), tr87 (127)."""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_agi3_world_model import grid_of, objects
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed

GAME = sys.argv[1] if len(sys.argv) > 1 else "ka59"


def _mh():
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _ascii(g):
    return "\n".join("".join(str(int(v))[-1] for v in row) for row in np.asarray(g))


def _explore_step(env, f, g, rng):
    """One action-aware exploration step: prefer clicking an object (config games respond to clicks on
    components, not bare cells) when action 6 is offered, else a non-noop keyboard key. Returns the new
    frame (or None if the env rejected the action)."""
    av = list(getattr(f, "available_actions", []) or [])
    if 6 in av and objects(g):
        oy, ox = objects(g)[rng.randrange(len(objects(g)))]
        return env.step(_game_action(GameAction, 6), data={"x": int(ox), "y": int(oy)})
    kb = [x for x in av if x not in (0, 6)] or [1, 2, 3, 4]
    return env.step(_game_action(GameAction, rng.choice(kb)), data=None)


def collect():
    """Return (scene grid, editable mask, win grid, non-win grids). Action-aware exploration discovers
    the editable region (the cells the player can change) and a sample of NON-win configurations."""
    arc = kit.offline_arcade(); env = arc.make(GAME, scorecard_id=arc.open_scorecard()); f = env.reset()
    scene = np.asarray(grid_of(f)).copy()
    rng = random.Random(0); ch = np.zeros_like(scene, bool); nonwins = []
    g = scene
    for i in range(60):
        nf = _explore_step(env, f, g, rng)
        if nf is None:
            f = env.reset(); g = np.asarray(grid_of(f)); continue
        g1 = np.asarray(grid_of(nf))
        if g1.shape == g.shape:
            ch |= (g != g1)
            if i % 10 == 0 and _levels_completed(nf) == 0:
                nonwins.append(g1.copy())
        f = nf; g = g1
    # banked win config (the PRE-win grid: the configuration that triggers the level-up)
    mh = _mh(); src = mh.RESOLVED_ARTIFACTS.get(GAME, mh.GAME_ARTIFACTS.get(GAME))
    acts = [mh.normalize(a) for a in (mh.load_actions(src) or []) if mh.normalize(a)[0] is not None]
    arc2 = kit.offline_arcade(); env2 = arc2.make(GAME, scorecard_id=arc2.open_scorecard()); f2 = env2.reset()
    win = None
    for aid, data in acts:
        prev = np.asarray(grid_of(f2))
        f2 = env2.step(_game_action(GameAction, aid), data=data)
        if f2 is None:
            break
        if _levels_completed(f2) > 0:
            win = prev; break
    return scene, ch, win, nonwins


def build_prompt(scene, ch):
    rs = np.argwhere(ch)
    r0, r1, c0, c1 = int(rs[:, 0].min()), int(rs[:, 0].max()), int(rs[:, 1].min()), int(rs[:, 1].max())
    return f"""You are reading an ARC-AGI-3 configuration puzzle ('{GAME}'). The state is a 64x64 grid of
integer colours (one digit per cell shown below; the digit is the colour mod 10). The player edits a
SELECTED region (clicking/arrow keys cycle the values of the EDITABLE cells). The level COMPLETES when
the editable configuration is CORRECT with respect to a rule that is VISIBLE elsewhere in the scene (a
reference pattern, a legend, or a rewrite-rule mapping).

EDITABLE region: rows {r0}..{r1}, cols {c0}..{c1} (the cells the player edits).
The rest of the grid contains the rule/reference the editable region must satisfy.

GRID (row 0 at top):
{_ascii(scene)}

Infer the WIN RULE from the visible structure and write a Python predicate:

    import numpy as np
    def is_win(grid):
        # grid: 64x64 numpy int array. Return True iff the EDITABLE region satisfies the win rule
        # (e.g. its colours equal/encode the reference shown elsewhere in the grid).
        ...

Use ONLY numpy + stdlib and the visible grid structure. Make is_win deterministic.

Return ONLY one ```python code block with def is_win.
```python
"""


def _generate_bounded(proposer, prompt, n_predict=1100, tries=2):
    """Bounded completion against the proposer's iGPU llama-server. The iGPU runs ~4.2 tok/s, so the
    proposer default (n_predict=4096) needs ~975s and trips the 900s timeout. We cap n_predict at 1100
    (~260s, safely bounded). NO ``` stop sequence: gemma re-emits ```python at the start of its output,
    which a ``` stop matches immediately (empty output). We capture the raw text and extract the code
    block. Returns (ok, code_or_msg, raw_sample) -- raw_sample lets the artifact self-document what the
    model actually produced (e.g. the degenerate comment-rambling that proves a scene-reading limit, not
    a plumbing bug). Reuses the proposer's server lifecycle (_ensure_server/stop)."""
    import ast
    import json as _json
    import urllib.request
    from carnot.agentic.arc_executable_world_model import _extract_python
    if not proposer._ensure_server():
        return False, f"GPU llama-server failed for {proposer.repo_substr} (no CPU fallback)", ""
    last = ""; raw = ""
    for attempt in range(tries):
        body = _json.dumps({"prompt": prompt, "n_predict": n_predict,
                            "temperature": 0.2 + 0.1 * attempt, "cache_prompt": True}).encode()
        try:
            req = urllib.request.Request(proposer._url() + "/completion", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=proposer.timeout) as r:
                raw = _json.load(r).get("content", "")
        except Exception as e:
            return False, f"local gguf (GPU server) failed: {e!r}"[:200], raw
        code = _extract_python(raw) or raw
        if "def is_win" not in code:
            last = "no def is_win in output (model rambled, never wrote the predicate)"; continue
        try:
            ast.parse(code)
        except SyntaxError as se:
            last = f"syntax error line {se.lineno}: {se.msg}"; continue
        return True, code, raw
    return False, f"local model code unusable after {tries} tries ({last})", raw


def verify(is_win, win, nonwins):
    res = {"fires_on_win": None, "false_positive_rate": None, "n_nonwin": len(nonwins)}
    try:
        res["fires_on_win"] = bool(is_win(np.asarray(win))) if win is not None else None
    except Exception as ex:
        res["win_error"] = f"{type(ex).__name__}: {str(ex)[:100]}"
    fp = 0
    for nw in nonwins:
        try:
            if bool(is_win(np.asarray(nw))):
                fp += 1
        except Exception:
            pass
    res["false_positive_rate"] = round(fp / max(1, len(nonwins)), 3)
    return res


def main():
    print(f"== LAYER B LLM-scene-reader on {GAME} (simpler-rule-class test vs tr87) ==", flush=True)
    scene, ch, win, nonwins = collect()
    print(f"  scene 64x64 | editable={int(ch.sum())} cells | win_config={'yes' if win is not None else 'NO'} | "
          f"non-wins={len(nonwins)}", flush=True)
    if win is None or not ch.any():
        out = {"experiment": "arc3_config_layerb_llm", "game": GAME,
               "honest_verdict": "complete_layerb_blocked_no_banked_win_or_no_editable_region",
               "inference_substrate": "offline_arc_agi3_layerb_llm_scene_reader_iGPU_port8920"}
        (REPO / "results" / f"arc3_config_layerb_llm_{GAME}.json").write_text(json.dumps(out, indent=2, default=str))
        print(f"  -> {out['honest_verdict']}", flush=True)
        return 0
    # iGPU is ~10x slower than a 3090. The proposer default (n_predict=4096, NO stop) runs PAST 900s on
    # the iGPU because the model generates to max_tokens. We bound it: the prompt primes ```python so the
    # model starts coding immediately; cap n_predict low and STOP at the closing fence. An is_win
    # predicate is short (~300-600 tokens), so 1200 + a fence-stop captures it in a fraction of the time.
    proposer = e3.LocalGGUFProposer(repo_substr="gemma-4-12B-it", port=8920, timeout=600)
    out = {"experiment": "arc3_config_layerb_llm", "game": GAME,
           "editable_cells": int(ch.sum()), "win_editable_colors": sorted(set(int(win[r, c]) for r, c in np.argwhere(ch))),
           "generation": {"n_predict": 1100, "stop": None, "bounded": True, "igpu_tok_per_s": 4.2},
           "inference_substrate": "offline_arc_agi3_layerb_llm_scene_reader_iGPU_port8920"}
    try:
        ok, code, raw = _generate_bounded(proposer, build_prompt(scene, ch))
        out["llm_proposed_rule"] = bool(ok)
        out["raw_sample"] = str(raw)[:800]  # self-documents what the model actually produced
        if not ok:
            out["honest_verdict"] = "complete_layerb_llm_failed_to_propose_rule_local_gguf_insufficient"
            out["msg"] = str(code)[:200]
        else:
            (REPO / "results" / "arc_config_layerb").mkdir(parents=True, exist_ok=True)
            p = REPO / "results" / "arc_config_layerb" / f"{GAME}_is_win.py"
            p.write_text(code)
            ns = {}
            try:
                exec(code, ns)  # noqa: S102 -- inducing a verifier predicate from the LLM; sandboxed-by-scope
                v = verify(ns.get("is_win", lambda g: False), win, nonwins)
                out["verification"] = v
                grounded = bool(v.get("fires_on_win")) and (v.get("false_positive_rate") or 1.0) < 0.2
                out["rule_grounded"] = grounded
                out["honest_verdict"] = ("complete_layerb_llm_induced_rule_GROUNDED" if grounded else
                                         "complete_layerb_llm_induced_rule_NOT_grounded_fails_verification")
            except Exception as ex:
                out["exec_error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
                out["honest_verdict"] = "complete_layerb_llm_rule_uncompilable"
    finally:
        try:
            proposer.stop()
        except Exception:
            pass
    (REPO / "results" / f"arc3_config_layerb_llm_{GAME}.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"  proposed={out.get('llm_proposed_rule')} verification={out.get('verification')} "
          f"grounded={out.get('rule_grounded')}", flush=True)
    print(f"  -> {out.get('honest_verdict')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
