"""BLAST RADIUS A/B: what does ADMITTING a weak (~0.2-cell_recall) engine cost?

PAIRED design, per game, identical seed and identical injected engine. The ONLY
difference between arms is whether the trust gate ADMITS that engine:

  reject  -- the SHIPPED gate (`_gate_value < 0.5`). The engine is rejected, no plan
             installs, the agent keeps exploring. This is today's behaviour.
  admit   -- the gate threshold emulated at ~0.19 on cell_recall (the proposed
             relaxation). The same engine is admitted, its plan installs, the agent
             executes it.

WHY A THRESHOLD EMULATION RATHER THAN AN ENV FLAG. The plain-branch threshold is a
HARDCODED literal `0.5` in `E3AgentPolicy._induce_and_plan`; `CARNOT_ARC_TRUST_METRIC`
switches WHICH quantity is compared but not what it is compared against. There is no
threshold knob to set. So the admit arm wraps `WorldModelVerifier` with a subclass that
reports `cell_recall * (0.5/0.19)` -- a monotone rescaling whose ONLY effect is to move
the decision boundary to 0.19. Every metric RECORDED here is the TRUE, unscaled value,
captured before scaling. Nothing on disk and no shipped default is modified; both
patches live in this process only and are restored in a `finally`.

INJECTION. `_induce_and_plan` gets its engine from `self._proposer().induce(...)`
(which WRITES `results/arc_e3/<game>/world_model.py`) followed by `e3.load_engine(...)`
(which reads it back). `results/arc_e3` is EVIDENCE: read, never write. So the stub
proposer's `induce` writes NOTHING and returns ok=True, and `load_engine` is patched to
hand back the injected engine. No evidence file is touched by either arm.

THE INJECTED ENGINE is a PARTIAL-MEMORISATION model, the realistic shape of a weak
induced world model: it memorises the agent's own observed transitions but reproduces
only a fraction `f` of each transition's changed cells (the rest keep their OLD value,
i.e. are wrong), and it generalises not at all -- an unseen (state, action) drifts by a
fixed roll. `cell_recall` is by construction ~= f, so setting f just above the proposed
0.19 threshold produces exactly the class of engine that relaxation would newly admit.
Its win predicate is pinned to the grid this engine itself reaches after `PLAN_K` steps,
so `plan_in_model` is GUARANTEED to find a plan. That is the worst case the task asks
for: a model wrong about most of the board AND confident it has found a win.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from typing import Any

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
sys.path.insert(0, REPO + "/python")

PLAN_K = 12  # induced plan depth, fixed so the blast radius is a controlled quantity
ADMIT_THRESHOLD = 0.19
TARGET_F = 0.30  # calibrated on tu93: true cell_recall ~0.26, inside the 0.19-0.28 band


def _key(g) -> bytes:
    import numpy as np

    return hashlib.sha256(np.asarray(g).astype("int16").tobytes()).digest()


def build_weak_engine(transitions, root_grid, f: float = TARGET_F, seed: int = 7):
    """A partial-memorisation engine with cell_recall ~= f, plus a reachable win."""
    import numpy as np

    rng = np.random.RandomState(seed)
    table: dict[tuple[bytes, int], Any] = {}
    for t in transitions:
        g0 = np.asarray(t.grid)
        g1 = np.asarray(t.next_grid)
        if g0.shape != g1.shape:
            continue
        m = g0 != g1
        pred = g0.copy()
        idx = np.argwhere(m)
        if len(idx):
            k = int(round(f * len(idx)))
            if k:
                pick = rng.choice(len(idx), size=k, replace=False)
                for j in pick:
                    r, c = idx[j]
                    pred[r, c] = g1[r, c]
        table[(_key(g0), int(t.action))] = pred

    def engine(grid, action, data=None):
        g = np.asarray(grid)
        hit = table.get((_key(g), int(action)))
        if hit is not None and hit.shape == g.shape:
            return hit.copy()
        return np.roll(g, 1, axis=0)  # generalises not at all: a fixed drift

    g = np.asarray(root_grid)
    seq = [(i % 4) + 1 for i in range(PLAN_K)]
    for a in seq:
        g = np.asarray(engine(g, a, None))
    target = _key(g)
    reachable = target != _key(np.asarray(root_grid))

    def is_level_complete(grid) -> bool:
        return _key(grid) == target

    return engine, is_level_complete, reachable


class StubProposer:
    """Stands in for LocalGGUFProposer. Writes NOTHING (no LLM, no evidence write)."""

    def __init__(self) -> None:
        self.include_playbook_exemplars = False
        self.no_think_prefix = ""
        self.max_tokens = 0
        self.tries = 1
        self.calls = 0

    def induce(self, game, transitions, cell, **kw):
        self.calls += 1
        return True, ""

    def liveness_witness(self) -> dict:
        return {"stub": True, "calls": self.calls}


def run(
    game: str,
    arm: str,
    *,
    seed: int = 20260802,
    budget: int = 140,
    explore_budget: int = 24,
    max_inductions: int = 3,
) -> dict:
    import os
    import random

    import numpy as np
    from arcengine import GameAction
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of

    random.seed(seed)
    np.random.seed(seed)

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    latest = env.reset()
    frames = [latest]
    prop = StubProposer()
    pol = E3AgentPolicy(game, proposer=prop, explore_budget=explore_budget)

    true_scores: list[dict] = []
    real_verifier = e3.WorldModelVerifier

    class _Verifier(real_verifier):  # type: ignore[misc,valid-type]
        def score(self, engine, max_mismatch: int = 8):
            vr = super().score(engine, max_mismatch=max_mismatch)
            true_scores.append(
                {
                    "accuracy": round(float(vr.accuracy), 6),
                    "cell_recall": round(float(vr.cell_recall), 6),
                    "n": int(vr.n),
                    "n_changing": int(vr.n_changing),
                    "correct_changed_cells": int(vr.correct_changed_cells),
                    "spurious_changed_cells": int(vr.spurious_changed_cells),
                }
            )
            if arm == "admit":
                import dataclasses

                vr = dataclasses.replace(
                    vr, cell_recall=min(1.0, float(vr.cell_recall) * (0.5 / ADMIT_THRESHOLD))
                )
            return vr

    build_log: list[dict] = []

    def _fake_load_engine(g: str):
        root = pol.root_grid
        if root is None:
            from carnot.agentic.arc_agi3_world_model import grid_of
            from carnot.agentic.arc_executable_world_model import detect_cell, to_logical

            root = to_logical(grid_of(latest), detect_cell(grid_of(latest)))
        tr = list(getattr(pol, "transitions", []) or [])
        eng, done, reach = build_weak_engine(tr, root)
        build_log.append({"n_transitions": len(tr), "win_reachable": bool(reach)})
        return eng, done

    os.environ["CARNOT_ARC_TRUST_METRIC"] = "cell_recall"
    e3.WorldModelVerifier = _Verifier  # type: ignore[misc]
    real_load = e3.load_engine
    e3.load_engine = _fake_load_engine  # type: ignore[assignment]

    t0 = time.time()
    actions = 0
    game_over_at: int | None = None
    actions_after_game_over = 0
    plan_installs: list[int] = []
    plan_step_actions = 0
    err = None
    hit_cap = False
    lvl0 = _level_of(latest)
    best_lvl = lvl0
    frames_seen: set = set()
    novel_frames = 0
    try:
        for _ in range(budget):
            if time.time() - t0 > 420:
                break
            if (
                len(getattr(pol, "induction_attempts", []) or []) >= max_inductions
                and getattr(pol, "phase", None) != "execute"
            ):
                hit_cap = True
                break
            try:
                if pol.is_done(frames, latest):
                    break
            except Exception:
                pass
            n_before = len(pol.plan)
            kind, data = pol.next_move(frames, latest)
            if len(pol.plan) and len(pol.plan) != n_before:
                plan_installs.append(len(pol.plan))
            if getattr(pol, "_prov_top", "") == "execute.plan_step":
                plan_step_actions += 1
            if kind is None:
                break
            if kind == "RESET":
                latest = env.reset()
            else:
                latest = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
            actions += 1
            frames.append(latest)
            try:
                from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of

                h = frame_hash(grid_of(latest))
                if h not in frames_seen:
                    frames_seen.add(h)
                    novel_frames += 1
            except Exception:
                pass
            st = str(getattr(latest, "state", ""))
            if "GAME_OVER" in st:
                if game_over_at is None:
                    game_over_at = actions
                else:
                    actions_after_game_over += 1
            lv = getattr(latest, "levels_completed", None)
            if lv is not None and int(lv) > best_lvl:
                best_lvl = int(lv)
    except Exception as exc:
        err = repr(exc)[:300]
    finally:
        e3.WorldModelVerifier = real_verifier  # type: ignore[misc]
        e3.load_engine = real_load  # type: ignore[assignment]
        os.environ.pop("CARNOT_ARC_TRUST_METRIC", None)

    ex = pol.explorer
    graph = getattr(ex, "graph", {}) or {}
    untested_left = sum(len(n.get("untested") or []) for n in graph.values())
    attempts = list(getattr(pol, "induction_attempts", []) or [])
    # The DECIDING reading, recovered by inverting the emulation scale (exact arithmetic).
    decided = []
    for a in attempts:
        v = a.get("verify_cell_recall")
        if v is None:
            decided.append(None)
        elif arm == "admit":
            decided.append(round(min(1.0, float(v)) * ADMIT_THRESHOLD / 0.5, 6))
        else:
            decided.append(round(float(v), 6))
    return {
        "game": game,
        "arm": arm,
        "seed": seed,
        "budget": budget,
        "actions": actions,
        "error": err,
        "hit_induction_cap": hit_cap,
        "n_induction_attempts": len(attempts),
        "n_planned": sum(1 for a in attempts if a.get("planned")),
        "skips": [a.get("skipped") for a in attempts],
        "deciding_true_cell_recall": decided,
        "true_gate_readings": true_scores,
        "engine_build_log": build_log,
        "plan_installs": plan_installs,
        "plan_step_actions": plan_step_actions,
        "level_start": lvl0,
        "level_best": best_lvl,
        "levels_gained": best_lvl - lvl0,
        "game_over_at_action": game_over_at,
        "actions_after_game_over": actions_after_game_over,
        "explored_out": bool(getattr(ex, "explored_out", False)),
        "transitions_collected": len(getattr(pol, "transitions", []) or []),
        "graph_nodes": len(graph),
        "frontier_untested_remaining": untested_left,
        "novel_frames": novel_frames,
        "wall_s": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    dest = sys.argv[1]
    out = []
    for g in sys.argv[2:]:
        for arm in ("reject", "admit"):
            try:
                out.append(run(g, arm))
            except Exception as exc:
                out.append({"game": g, "arm": arm, "fatal": repr(exc)[:400]})
            with open(dest, "w") as fh:
                json.dump(out, fh, indent=2)
    print("WROTE", dest, len(out))
