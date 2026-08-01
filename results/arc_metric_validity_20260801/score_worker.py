#!/usr/bin/env python3
"""Score ONE induced world model on BOTH sides of the question, in a killable process.

THE QUESTION. `change_fidelity` is the primary the object-perception A/B moved. Its only
consumer is `plan_in_model`, which walks the engine FORWARD from a level root. So the metric is
worth optimising only if a higher value makes a plan more likely to exist. This worker produces,
per engine, the metric on one side and plannability on the other, plus every rival predictor the
review named, so the association can be estimated and so a rival can beat it.

NOTHING GENERATED IS EXECUTED IN THE DRIVER'S INTERPRETER. On 2026-07-31 a non-terminating
induced engine (ft09 candidate 5) wedged a generation loop for 13 minutes. An in-process SIGALRM
cannot fix that here, because `plan_in_model`, `_goal_satisfiability_check`,
`WorldModelVerifier.score` and `dry_run_defects` ALL wrap the engine call in a broad
`except Exception: continue` -- the alarm's exception is swallowed by the very code meant to be
interrupted, and a hang becomes a silent clean zero. Only a separate process can be killed.

THE BUDGETS ARE THE SHIPPED ONES, DELIBERATELY. `_goal_satisfiability_check` is called with no
`max_nodes`/`max_depth` so it reads `_goal_gate_max_nodes_default()` and
`e3.plan_max_depth_default()`; `plan_in_model` likewise. Both therefore run at max_depth 80, the
default since 2026-07-31. The frozen best-of-N candidates were scored at the PRIOR default of 40
(`goal_max_depth: 40` is on every one of its 48 records), so re-planning them here is not a
duplicate -- it is the same question asked at the horizon the live agent now uses, and the
resolver's own docstring records that the change converts 2 candidates into 6 on that corpus.

`goal_energy` is NOT supplied, matching the frozen run. The live stall path may install a
best-first heuristic, which can only reach a goal in FEWER nodes, so plain FIFO BFS makes every
`plan_found` here a LOWER bound on what the live planner could find. That is the conservative
direction for a run whose headline may be "the metric predicts nothing".
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import sys
import time

REPO = pathlib.Path(os.environ["CARNOT_REPO"])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_metric_validity/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, str(REPO / "python"))

STATE_PROBE_MAX_CALLS = 600


def _state_graph_probe(engine, root, e3, np) -> dict:
    """Is the engine's reachable state graph a TREE or a PATH?

    WHY THIS IS A FIRST-CLASS PREDICTOR AND NOT A CURIOSITY. `plan_max_depth_default`'s own
    measurement records that tn36's 32 changing root actions collapse to ONE distinct successor
    -- the engine fills the next cell wherever the click lands -- and both search sites dedup by
    state key, so the search tree degenerates to a PATH. On a path the node budget buys depth
    one-for-one and only `max_depth` can stop the search, which is why raising the horizon and
    not `max_nodes` was the correct fix there. That makes branching a structural determinant of
    whether a plan is findable at all, entirely separate from how accurate the engine is.

    Bounded at `STATE_PROBE_MAX_CALLS` engine calls, far below the planner's 20000, so this is a
    cheap shape probe and not a second search.
    """
    out: dict = {}
    try:
        root = np.asarray(root)
        seen = {e3._state_key(root)}  # noqa: SLF001
        frontier = [root]
        calls = 0
        expanded = 0
        depth = 0
        new_per_expansion: list[int] = []
        root_successors: set = set()
        root_changing_successors: set = set()
        root_key = e3._state_key(root)  # noqa: SLF001
        while frontier and calls < STATE_PROBE_MAX_CALLS:
            nxt = []
            for grid in frontier:
                if calls >= STATE_PROBE_MAX_CALLS:
                    break
                expanded += 1
                fresh = 0
                for c in e3._model_candidates(grid):  # noqa: SLF001
                    if calls >= STATE_PROBE_MAX_CALLS:
                        break
                    try:
                        ng = np.asarray(engine(grid.copy(), c["action"], c["data"]))
                    except Exception:  # noqa: BLE001
                        continue
                    calls += 1
                    if ng.shape != root.shape:
                        continue
                    k = e3._state_key(ng)  # noqa: SLF001
                    if depth == 0:
                        root_successors.add(k)
                        if k != root_key:
                            root_changing_successors.add(k)
                    if k in seen:
                        continue
                    seen.add(k)
                    fresh += 1
                    nxt.append(ng)
                new_per_expansion.append(fresh)
            frontier = nxt
            depth += 1
        out["probe_engine_calls"] = calls
        out["probe_nodes_expanded"] = expanded
        out["probe_distinct_states"] = len(seen)
        out["probe_depth_reached"] = depth
        # The headline shape number: how many NEW states an expansion yields on average. ~1.0 is
        # a path (depth-bound), >>1 is a tree (breadth-bound), 0.0 is inert.
        out["probe_mean_new_states_per_expansion"] = (
            round(sum(new_per_expansion) / len(new_per_expansion), 4) if new_per_expansion else None
        )
        out["n_distinct_successors_at_root"] = len(root_successors)
        out["n_distinct_changing_successors_at_root"] = len(root_changing_successors)
        out["engine_changes_anything_at_root"] = bool(root_changing_successors)
        out["probe_status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["probe_status"] = f"raised:{type(exc).__name__}"
        out["probe_error"] = str(exc)[:200]
    return out


def _score_split(vr) -> dict:
    """Copy the VerifyResult channels. Fields, not compute -- every one comes off one score()."""
    return {
        "n": int(vr.n),
        "n_correct": int(vr.n_correct),
        "accuracy": round(float(vr.accuracy), 6),
        "cell_recall": round(float(vr.cell_recall), 6),
        "n_changing": int(vr.n_changing),
        "n_changes_correct": int(vr.n_changes_correct),
        "change_accuracy": round(float(vr.change_accuracy), 6),
        "change_fidelity": round(float(vr.change_fidelity), 6),
        "correct_changed_cells": int(vr.correct_changed_cells),
        "spurious_changed_cells": int(vr.spurious_changed_cells),
        "invented_changed_cells": int(vr.invented_changed_cells),
        "n_noop": int(vr.n_noop),
        "n_noop_hallucinated": int(vr.n_noop_hallucinated),
        "noop_channel_measurable": bool(vr.noop_channel_measurable),
        "hud_mask_status": str(vr.hud_mask_status),
    }


def main() -> int:  # noqa: C901, PLR0912, PLR0915
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    out: dict = {"status": "ok", "cell": job["cell"], "corpus": job["corpus"], "game": job["game"]}

    import numpy as np

    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_llm_reinduction import _goal_satisfiability_check
    from carnot.agentic.arc_world_model_trust_energy import (
        WorldModelCandidate,
        select_trusted_world_model,
    )

    with open(job["window_pkl"], "rb") as fh:
        win = pickle.load(fh)
    shown, held = list(win["shown"]), list(win["held"])

    code_path = pathlib.Path(job["code_path"])
    if not code_path.exists():
        out["status"] = "no_engine_file"
        print(json.dumps(out))
        return 0
    src = code_path.read_text()

    ns: dict = {"np": np, "numpy": np}
    try:
        exec(compile(src, str(code_path), "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"unrunnable:{type(exc).__name__}"
        out["error"] = str(exc)[:200]
        print(json.dumps(out))
        return 0
    engine = ns.get("engine")
    goal = ns.get("is_level_complete")
    out["has_engine"] = callable(engine)
    out["has_goal_predicate"] = callable(goal)
    if not callable(engine):
        out["status"] = "no_engine_symbol"
        print(json.dumps(out))
        return 0

    # ---- THE METRIC SIDE: held-out change_fidelity, plus in-sample as a rival ---------------
    t = time.monotonic()
    out["heldout"] = _score_split(e3.WorldModelVerifier(list(held)).score(engine))
    out["in_sample"] = _score_split(e3.WorldModelVerifier(list(shown)).score(engine))
    out["score_wall_s"] = round(time.monotonic() - t, 2)

    # ---- the shipped trust gate, on the transitions induce() actually saw -------------------
    try:
        sel = select_trusted_world_model(
            list(shown),
            [WorldModelCandidate("cell_candidate", engine, goal if callable(goal) else None)],
            hidden_state=True,
        )
        out["shipped_gate_heldout_accuracy"] = round(float(sel.selected_score.heldout_accuracy), 6)
        out["shipped_gate_prefix_accuracy"] = round(float(sel.selected_score.prefix_accuracy), 6)
        out["shipped_gate_trust_energy"] = round(float(sel.selected_score.trust_energy), 6)
        out["shipped_gate_passes"] = bool(sel.selected_score.heldout_accuracy >= 1.0)
    except Exception as exc:  # noqa: BLE001
        out["shipped_gate_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    # ---- THE PLANNABILITY SIDE, from every available root -----------------------------------
    roots: dict[str, np.ndarray] = {
        "window_root": np.asarray(win["window_root"]),
        "held_root": np.asarray(win["held_root"]),
    }
    rp = job.get("real_root_pkl")
    if rp and os.path.exists(rp):
        with open(rp, "rb") as fh:
            roots["real_root"] = np.asarray(pickle.load(fh))

    out["roots_available"] = sorted(roots)
    out["plan"] = {}
    out["goal_gate"] = {}
    out["state_graph"] = {}

    for name, root in roots.items():
        # the engine's own state-graph shape from this root (cheap, bounded)
        out["state_graph"][name] = _state_graph_probe(engine, root, e3, np)

        # criterion (ii): the goal gate at the SHIPPED budget
        t = time.monotonic()
        g: dict = {}
        try:
            check = _goal_satisfiability_check(
                engine=engine, goal=goal if callable(goal) else None, start_grid=root
            )
            g["satisfiable"] = bool(check.get("satisfiable"))
            g["kind"] = str((check.get("counterexample") or {}).get("kind") or "satisfiable")
            for k in (
                "first_true_depth",
                "reachable_grids_evaluated",
                "engine_calls",
                "engine_errors",
                "max_nodes",
                "max_depth",
                "termination",
                "frontier_remaining",
                "depth_truncated_nodes",
            ):
                if k in check:
                    g[k] = check[k]
        except Exception as exc:  # noqa: BLE001
            g["satisfiable"] = False
            g["kind"] = f"gate_raised:{type(exc).__name__}"
            g["error"] = str(exc)[:200]
        g["wall_s"] = round(time.monotonic() - t, 2)
        out["goal_gate"][name] = g

        # criterion (iii): the planner at the SHIPPED budget
        t = time.monotonic()
        diag: dict = {}
        p: dict = {}
        try:
            plan = e3.plan_in_model(
                engine, goal if callable(goal) else None, root, diagnostics=diag
            )
            p["plan_found"] = bool(plan)
            p["plan_length"] = len(plan) if plan else 0
        except Exception as exc:  # noqa: BLE001
            p["plan_found"] = False
            p["plan_length"] = 0
            p["plan_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        p["diagnostics"] = {
            k: v for k, v in diag.items() if isinstance(v, (int, float, str, bool))
        }
        p["wall_s"] = round(time.monotonic() - t, 2)
        out["plan"][name] = p

    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
