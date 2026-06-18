"""Config rule-inducer LAYER B -- the LLM-scene-reader, first test on tr87 (KNOWN rule = glyph-rewrite).

Plan: docs/research-notes/arc-config-target-induction-scope-2026-06-18.md (Section 4.1, corrected
direction). Layer A (interaction vocabulary) is done; Layer B is the WIN-RULE -- the relation the
configuration must satisfy, which no generic heuristic captures. This tests whether a LOCAL open-weight
GGUF (offline-legal per Decentralization Rule 1) can READ the tr87 scene and INDUCE the win-rule.

The Carnot propose-then-ground pattern: the LLM PROPOSES the rule as a checkable Python predicate
`is_win(grid)`; the VERIFIER grounds it -- the induced rule must fire True on the banked WIN config and
False on NON-win configs. tr87 is the ground-truth case (its rule is the glyph-rewrite solved earlier by
a GameAdapter), so we can score the induced rule against reality.

Honest first test: if gemma-12B cannot read this complex puzzle, that is a finding (local GGUF
insufficient for scene-level rule induction -> a stronger reader, or a more structured inducer, is
needed -- the known decentralization tradeoff). iGPU port 8920; zero quota; tears down its server."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed

GAME = "tr87"


def _mh():
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _ascii(g):
    return "\n".join("".join(str(int(v))[-1] for v in row) for row in np.asarray(g))


def collect():
    """Return (scene grid, editable mask, win grid, non-win grids, sample transitions)."""
    import random
    arc = kit.offline_arcade(); env = arc.make(GAME, scorecard_id=arc.open_scorecard()); f = env.reset()
    scene = np.asarray(grid_of(f)).copy()
    rng = random.Random(0); ch = np.zeros_like(scene, bool); nonwins = []; trans = []
    g = scene
    for i in range(40):
        a = rng.choice([1, 2, 3, 4])
        nf = env.step(_game_action(GameAction, a), data=None)
        if nf is None:
            f = env.reset(); g = np.asarray(grid_of(f)); continue
        g1 = np.asarray(grid_of(nf))
        if g1.shape == g.shape:
            ch |= (g != g1)
            if i < 2:
                trans.append((a, np.argwhere(g != g1)[:6].tolist()))
            if i % 8 == 0:
                nonwins.append(g1.copy())
        g = g1
    # banked win config (pre-win)
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
    return scene, ch, win, nonwins, trans


def build_prompt(scene, ch):
    rs = np.argwhere(ch)
    r0, r1, c0, c1 = int(rs[:, 0].min()), int(rs[:, 0].max()), int(rs[:, 1].min()), int(rs[:, 1].max())
    return f"""You are reading an ARC-AGI-3 configuration puzzle ('{GAME}'). The state is a 64x64 grid of
integer colours (one digit per cell shown below; the digit is the colour mod 10). Pressing arrow keys
cycles the value of a SELECTED glyph in the EDITABLE region. The level COMPLETES when the editable
configuration is CORRECT with respect to a rule that is VISIBLE elsewhere in the scene (a reference
pattern or a rewrite-rule mapping).

EDITABLE region: rows {r0}..{r1}, cols {c0}..{c1} (the glyphs the player edits).
The rest of the grid contains the rule/reference the editable region must satisfy.

GRID (row 0 at top):
{_ascii(scene)}

Infer the WIN RULE from the visible structure and write a Python predicate:

    import numpy as np
    def is_win(grid):
        # grid: 64x64 numpy int array. Return True iff the EDITABLE region satisfies the win rule
        # (e.g. its glyph sequence equals the rewrite/reference shown elsewhere in the grid).
        ...

Use ONLY numpy + stdlib and the visible grid structure. Make is_win deterministic.

Return ONLY one ```python code block with def is_win.
```python
"""


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
    print(f"== LAYER B LLM-scene-reader on {GAME} (known rule = glyph-rewrite) ==", flush=True)
    scene, ch, win, nonwins, trans = collect()
    print(f"  scene 64x64 | editable={int(ch.sum())} cells | win_config={'yes' if win is not None else 'NO'} | "
          f"non-wins={len(nonwins)}", flush=True)
    # iGPU is ~10x slower than a 3090: a 4096-token rule needs >300s. 900s avoids a false timeout
    # (the prior gemma-on-iGPU experiments confirmed the default 300s cuts off generation mid-rule).
    proposer = e3.LocalGGUFProposer(repo_substr="gemma-4-12B-it", port=8920, timeout=900)
    out = {"experiment": "arc3_config_layerb_llm_tr87", "game": GAME,
           "inference_substrate": "offline_arc_agi3_layerb_llm_scene_reader_iGPU_port8920"}
    try:
        ok, code = proposer.generate(build_prompt(scene, ch), required=("is_win",))
        out["llm_proposed_rule"] = bool(ok)
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
    (REPO / "results" / "arc3_config_layerb_llm_tr87.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"  proposed={out.get('llm_proposed_rule')} verification={out.get('verification')} "
          f"grounded={out.get('rule_grounded')}", flush=True)
    print(f"  -> {out.get('honest_verdict')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
