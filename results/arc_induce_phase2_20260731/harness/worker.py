#!/usr/bin/env python3
"""PHASE 2, worker -- score ONE induced candidate for the two Phase-2 questions, in isolation.

WHY A SUBPROCESS, ALWAYS. Identical reasoning to the Phase-1 `gate_worker.py`, and it is not
hypothetical here either. On 2026-07-31 the Phase-1 GENERATION loop wedged for 13 minutes at 32%
CPU with both GPUs idle, spinning inside `validate_engine_code`'s dry run of a non-terminating
induced engine (ft09 candidate 5). Everything this file does -- the rollout and the
change-prediction pass -- calls the generated `engine()` in a loop, so a single non-terminating
engine would hang the driver exactly the same way. A signal-based in-process alarm cannot save
us: the production helpers wrap engine invocation in broad `except Exception: continue`, so an
in-process timeout would be SWALLOWED and become a silent false CLEAN. So the boundary is the
process boundary, and the driver enforces it with a hard `timeout=`.

WHAT THIS MEASURES

(2A) DEPTH-TO-GOAL, BY CONSTRUCTIVE GREEDY ROLLOUT. The shipped goal gate
(`_goal_satisfiability_check`) is a breadth-first search bounded at `max_depth=40` /
`max_nodes=20000`. With ~37 successors per state, BFS cannot physically reach depth 40, let
alone past it -- so a goal that is genuinely reachable but DISTANT is reported as
`goal_unreached_within_depth`, which is an UNDECIDED verdict, not a disproof. This rollout
answers the question BFS cannot afford to: walking the engine's own model greedily -- always
step to a predicted-changing, not-yet-seen state -- at what depth does the engine's own
`is_level_complete` first become true? Cost is LINEAR in depth (one candidate expansion per
step) instead of exponential, so depths in the hundreds are affordable.

  * A depth <= 40 means the shipped gate could in principle have seen it.
  * A depth in 41..MAX_STEPS means the goal is REACHABLE AND THE GATE CANNOT SEE IT. That is a
    search-horizon artifact, not a wrong predicate.
  * No goal-true within MAX_STEPS is evidence for (not proof of) a genuinely unreachable or
    degenerate predicate.

This rollout is DIAGNOSTIC. It does not widen, disable, or replace the shipped gate, and a
depth found here is NEVER counted as a criterion (ii)/(iii) pass -- the Phase-1 yields stand
exactly as reported. It only tells us WHY the gate answered as it did.

Note the rollout is also, by construction, exactly the decoupled explorer of (2C): it selects
actions using ONLY the engine's dynamics (step toward predicted change, avoid predicted
revisits) and never consults the goal to choose. The goal is read as a passive observer.

(2C) CHANGE PREDICTION, OUT OF SAMPLE. For each held-out transition, does the engine correctly
predict WHETHER the grid changes (not what it changes to)? This is the weakest useful signal a
dynamics model can offer an explorer, and unlike a plan it needs no certified goal. Reported as
a full confusion matrix so the driver can compute balanced accuracy and, more importantly, the
INCREMENTAL value over the dedup the explorer already performs.

The engine is called with a COPY of the grid every time. Induced engines routinely mutate their
argument in place and return it (tn36 candidate 1 does exactly `grid[1, col] = 3; return grid`),
so passing the live grid would corrupt the rollout state and silently fabricate "changes".
"""

from __future__ import annotations

import json
import os
import pickle
import sys
import time

REPO = "/home/ianblenke/github.com/ianblenke/carnot"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_phase2_worker/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "python"))

MAX_STEPS = 400  # the live MAX_ACTIONS budget; a plan longer than this cannot bank a level
WALL_BUDGET_S = 150.0


def _load(path: str):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main() -> int:  # noqa: C901
    job = json.loads(open(sys.argv[1]).read())
    out: dict = {"status": "ok"}

    import numpy as np

    from carnot.agentic.arc_executable_world_model import _model_candidates, _state_key

    code = open(job["code_path"]).read()
    ns: dict = {"np": np, "numpy": np}
    try:
        exec(compile(code, job["code_path"], "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"unrunnable:{type(exc).__name__}"
        print(json.dumps(out))
        return 0
    engine = ns.get("engine")
    goal = ns.get("is_level_complete")
    out["has_engine"] = callable(engine)
    out["has_goal_predicate"] = callable(goal)
    if not callable(engine):
        out["status"] = "no_engine"
        print(json.dumps(out))
        return 0

    def _key(g) -> str:
        k = _state_key(np.asarray(g))
        return k.hex() if isinstance(k, bytes) else str(k)

    # ================= (2C) change prediction, out of sample =============================
    # Emitted per row (not just aggregated) so the driver can restrict to the first-visit
    # subset -- the only rows on which the engine could add anything the explorer's existing
    # state-key dedup does not already have.
    rows_out: list[dict] = []
    heldout = _load(job["heldout_pkl"]) if os.path.exists(job["heldout_pkl"]) else []
    n_pred_err = 0
    for tr in heldout:
        grid = np.asarray(tr.grid)
        nxt = np.asarray(tr.next_grid)
        actual_change = not np.array_equal(nxt, grid)
        rec = {
            "action": int(tr.action),
            "data": tr.data if isinstance(tr.data, dict) else None,
            "state_key": _key(grid),
            "actual_change": bool(actual_change),
        }
        try:
            pred = engine(np.array(grid, copy=True), int(tr.action), tr.data)
            pred = np.asarray(pred)
            rec["pred_change"] = bool(not np.array_equal(pred, grid))
            rec["pred_exact"] = bool(np.array_equal(pred, nxt))
        except Exception as exc:  # noqa: BLE001
            # An engine error is a MISSING OBSERVATION for this row, never a "no change".
            n_pred_err += 1
            rec["pred_change"] = None
            rec["pred_exact"] = None
            rec["pred_error"] = type(exc).__name__
        rows_out.append(rec)
    out["heldout_rows"] = rows_out
    out["heldout_n"] = len(rows_out)
    out["heldout_pred_errors"] = n_pred_err

    # ================= (2A) constructive greedy depth-to-goal =============================
    root_path = job.get("root_pkl")
    if not root_path or not os.path.exists(root_path):
        out["rollout"] = {"status": "no_root_grid"}
        print(json.dumps(out))
        return 0

    root = np.asarray(_load(root_path))

    # ---- CONTROL: is an inert engine really inert, or is the PLANNER asking the wrong
    # questions? The first pass of this phase found most rollouts dying immediately with
    # `engine_predicts_no_change_from_any_action`. There are two very different explanations and
    # they have opposite consequences:
    #
    #   (a) the engine genuinely models nothing -- an induction failure; or
    #   (b) the engine responds only to the KIND of action it was induced on, and
    #       `_model_candidates` (connected-component CENTROIDS) proposes clicks at coordinates
    #       that never appear in the tape, so a coordinate-keyed engine correctly reports no-op.
    #
    # (b) would be an ACTION-SPACE MISMATCH between induction and planning: the engine is fine
    # and the search simply cannot address it. The two are separated by re-probing the same root
    # with the actions the tape actually contains. Both probes are one engine call per candidate
    # action, from the identical root, so the comparison is exact.
    probe: dict = {}
    for label, cand_source in (
        ("planner_candidates", "model"),
        ("taped_actions", "tape"),
    ):
        try:
            if cand_source == "model":
                cands = _model_candidates(root)
            else:
                seen_pairs, cands = set(), []
                for tr in _load(job["full_pkl"]):
                    d = tr.data if isinstance(tr.data, dict) else None
                    pk = (int(tr.action), d.get("x") if d else None, d.get("y") if d else None)
                    if pk in seen_pairs:
                        continue
                    seen_pairs.add(pk)
                    cands.append({"action": int(tr.action), "data": d})
        except Exception as exc:  # noqa: BLE001
            probe[label] = {"error": type(exc).__name__, "detail": str(exc)[:160]}
            continue
        # DISTINCT successors, not just changing ones. This is the EFFECTIVE BRANCHING FACTOR --
        # the quantity that decides whether the gate's node budget or its depth cap is the
        # binding constraint. Both `_goal_satisfiability_check` and `plan_in_model` dedup by
        # state key, so 32 different clicks that all drive the engine to the SAME next grid cost
        # one node between them, not 32. An engine whose successors collapse to a single state
        # makes the search a PATH, on which 20,000 nodes buys 20,000 depth and the only thing
        # that can stop the search is `max_depth`.
        n_change = n_err = 0
        succ: set[str] = set()
        for c in cands:
            try:
                g2 = np.asarray(engine(np.array(root, copy=True), int(c["action"]), c.get("data")))
                if g2.shape == root.shape and not np.array_equal(g2, root):
                    n_change += 1
                    succ.add(_key(g2))
            except Exception:  # noqa: BLE001
                n_err += 1
        probe[label] = {
            "n_candidates": len(cands),
            "n_predicted_changing": n_change,
            "n_distinct_successor_states": len(succ),
            "n_errors": n_err,
        }
    out["root_action_probe"] = probe

    grid = root
    roll: dict = {
        "max_steps": MAX_STEPS,
        "wall_budget_s": WALL_BUDGET_S,
        "goal_first_true_depth": None,
        "engine_errors": 0,
        "goal_errors": 0,
        "n_distinct_states": 1,
        "n_revisit_steps": 0,
        # The affordability currency. `arc_llm_reinduction` records production affordability as
        # ~17,854 engine calls per game, and reports that ka59's gate needed 160,000 and its
        # planner 137,347 nodes -- both unaffordable. Counting the same unit here makes the
        # greedy rollout directly comparable to those figures.
        "engine_calls": 0,
    }
    t0 = time.monotonic()

    def _goal_true(g) -> bool | None:
        if not callable(goal):
            return None
        try:
            return bool(goal(np.asarray(g)))
        except Exception:  # noqa: BLE001
            roll["goal_errors"] += 1
            return None

    if _goal_true(grid) is True:
        # Same semantics the shipped gate uses: true on the level's own opening screen is
        # degenerate, not satisfied.
        roll["status"] = "goal_true_at_root_degenerate"
        roll["depth_reached"] = 0
        out["rollout"] = roll
        print(json.dumps(out))
        return 0

    seen = {_key(grid)}
    depth = 0
    status = "max_steps_exhausted"
    while depth < MAX_STEPS:
        if time.monotonic() - t0 > WALL_BUDGET_S:
            status = "wall_budget_exhausted"
            break
        try:
            cands = _model_candidates(grid)
        except Exception:  # noqa: BLE001
            status = "candidate_generation_failed"
            break
        chosen = None
        fallback = None
        for c in cands:
            try:
                roll["engine_calls"] += 1
                g2 = np.asarray(engine(np.array(grid, copy=True), int(c["action"]), c.get("data")))
            except Exception:  # noqa: BLE001
                roll["engine_errors"] += 1
                continue
            if g2.shape != grid.shape or np.array_equal(g2, grid):
                continue  # engine predicts this action is a no-op
            if _key(g2) not in seen:
                chosen = g2
                break
            if fallback is None:
                fallback = g2
        if chosen is None:
            if fallback is None:
                status = "engine_predicts_no_change_from_any_action"
                break
            # Every changing action loops back to a state we have already stood on. Take it --
            # the engine may still be mid-cycle -- but record it, because a rollout dominated by
            # revisits is not making progress.
            chosen = fallback
            roll["n_revisit_steps"] += 1
        grid = chosen
        depth += 1
        k = _key(grid)
        if k not in seen:
            seen.add(k)
            roll["n_distinct_states"] += 1
        if _goal_true(grid) is True:
            roll["goal_first_true_depth"] = depth
            status = "goal_reached"
            break

    roll["status"] = status
    roll["depth_reached"] = depth
    roll["wall_s"] = round(time.monotonic() - t0, 2)
    out["rollout"] = roll
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
