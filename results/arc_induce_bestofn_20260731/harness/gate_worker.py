#!/usr/bin/env python3
"""BEST-OF-N, STEP 4a -- score ONE candidate end to end, in isolation. Every criterion.

WHY EVERY EXECUTION OF GENERATED CODE HAPPENS HERE, in a subprocess, and none of it in the
scorer.

This module originally covered only criteria (ii) and (iii) -- the goal gate and the planner --
on the reasoning that those two run generated code inside a search loop and so are the dangerous
ones. That reasoning was incomplete, and the run proved it. Criterion (i) also executes the
generated engine: `WorldModelVerifier.score(engine)` calls it once per transition, and
`select_trusted_world_model` calls it many more times. A generated engine that does not
terminate hangs the scorer exactly as readily as it hangs a search.

That is not hypothetical. On 2026-07-31 the GENERATION loop wedged for 13 minutes in state R at
32% CPU with no open socket and both GPUs idle, spinning inside `validate_engine_code`'s dry run
of a non-terminating induced engine (ft09 candidate 5). The same code would have been handed to
`WorldModelVerifier` moments later. So the boundary is drawn at the only place it can be drawn
safely: NO generated code is executed in the scorer's interpreter at all.

Two failure modes make in-process evaluation unsafe, and both are realistic here:

  * NON-TERMINATION. The search functions bound the SEARCH, not the engine: a single `engine()`
    call that loops forever is unbounded. Worse, `_goal_satisfiability_check`, `plan_in_model`
    and `dry_run_defects` all wrap engine invocation in a broad `except Exception: continue`, so
    a signal-based in-process alarm would be SWALLOWED by the very code it is meant to interrupt
    -- a hang would become a silent false CLEAN, which is worse than the hang.
  * MEMORY / INTERPRETER DAMAGE. Generated code was mechanically defective in 22 of 36 attempts
    in the preceding phase; an allocation blow-up or a C-level fault in a numpy call takes the
    whole scoring run with it if it shares the interpreter.

The cost of isolation is one interpreter start per candidate (~1s against a gate allowed
minutes), which is not worth optimising away.

WHAT IS AND IS NOT WIDENED. The gate and planner run at the SHIPPED defaults --
`_goal_satisfiability_check` with no `max_nodes` argument (so it reads
`_goal_gate_max_nodes_default()`, exactly as `execute_bounded_llm_reinduction` does) and
`plan_in_model` with its own defaults. Nothing is relaxed to manufacture a pass. A candidate that
fails because the budget ran out is recorded as `goal_unreached_within_budget` /
`goal_unreached_within_depth`, NOT as a degenerate predicate -- that distinction was fixed in the
repo on 2026-07-31 and flattening it here would reintroduce the mislabel one level up.

`goal_energy` is deliberately NOT supplied to `plan_in_model`. The live stall path calls
`self._guided_plan_in_model(e3.plan_in_model)`, which may install a goal-energy heuristic; using
the plain FIFO BFS here makes criterion (iii) a LOWER BOUND on what the live planner could find,
which is the conservative direction for a phase whose headline may be "(iii) yield is zero". A
best-first search can only reach the goal in FEWER nodes, never in more.
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
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_bon_gate/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, os.path.join(REPO, "python"))


def main() -> int:  # noqa: C901
    job = json.loads(open(sys.argv[1]).read())
    out: dict = {"status": "ok"}

    import numpy as np

    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_llm_reinduction import _goal_satisfiability_check
    from carnot.agentic.arc_world_model_trust_energy import (
        WorldModelCandidate,
        select_trusted_world_model,
    )

    code = open(job["code_path"]).read()
    ns: dict = {"np": np, "numpy": np}
    try:
        exec(compile(code, job["code_path"], "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"unrunnable:{type(exc).__name__}"
        out["error"] = str(exc)[:240]
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

    def _load(p):
        with open(p, "rb") as fh:
            return pickle.load(fh)

    # ---- criterion (i): held-out dynamics, against the PROVEN unrendered split ------------
    for label in ("heldout", "in_sample"):
        p = job.get(f"{label}_pkl")
        rows_ = _load(p) if p and os.path.exists(p) else []
        if not rows_:
            out[f"{label}_n"] = 0
            continue
        vr = e3.WorldModelVerifier(list(rows_)).score(engine)
        n_changing = int(getattr(vr, "n_changing", 0) or 0)
        out[f"{label}_n"] = int(getattr(vr, "n", 0) or 0)
        out[f"{label}_accuracy"] = round(float(getattr(vr, "accuracy", 0.0) or 0.0), 6)
        out[f"{label}_n_changing"] = n_changing
        out[f"{label}_n_changes_correct"] = int(getattr(vr, "n_changes_correct", 0) or 0)
        # None, not 0.0: with no changing rows this quantity was NOT measured.
        out[f"{label}_cell_recall"] = (
            round(float(getattr(vr, "cell_recall", 0.0) or 0.0), 4) if n_changing else None
        )
        out[f"{label}_invented_changed_cells"] = int(getattr(vr, "invented_changed_cells", 0) or 0)
        out[f"{label}_n_noop"] = int(getattr(vr, "n_noop", 0) or 0)
        out[f"{label}_n_noop_hallucinated"] = int(getattr(vr, "n_noop_hallucinated", 0) or 0)

    # ---- the SHIPPED trust gate, on the transitions induce() actually received -------------
    # Reported next to the out-of-sample bar because they are NOT the same measurement: this one
    # splits the 17-row prefix internally, the other uses the proven unrendered rows.
    try:
        prefix_trans = _load(job["prefix_pkl"])
        sel = select_trusted_world_model(
            list(prefix_trans),
            [WorldModelCandidate("bon_candidate", engine, goal if callable(goal) else None)],
            hidden_state=True,
        )
        out["shipped_gate_heldout_accuracy"] = round(float(sel.selected_score.heldout_accuracy), 6)
        out["shipped_gate_prefix_accuracy"] = round(float(sel.selected_score.prefix_accuracy), 6)
        out["shipped_gate_trust_energy"] = round(float(sel.selected_score.trust_energy), 6)
        out["shipped_gate_passes"] = bool(sel.selected_score.heldout_accuracy >= 1.0)
    except Exception as exc:  # noqa: BLE001
        out["shipped_gate_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"

    root_path = job.get("root_pkl")
    if not root_path or not os.path.exists(root_path):
        out["goal_kind"] = "no_root_grid_captured"
        print(json.dumps(out))
        return 0
    root = np.asarray(_load(root_path))

    # ---- criterion (ii): the goal gate, at the SHIPPED budget ------------------------------
    t = time.monotonic()
    try:
        check = _goal_satisfiability_check(
            engine=engine, goal=goal if callable(goal) else None, start_grid=root
        )
        out["goal_satisfiable"] = bool(check.get("satisfiable"))
        out["goal_kind"] = str((check.get("counterexample") or {}).get("kind") or "satisfiable")
        for k in (
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
                out[f"goal_{k}"] = check[k]
    except Exception as exc:  # noqa: BLE001
        out["goal_satisfiable"] = False
        out["goal_kind"] = f"gate_raised:{type(exc).__name__}"
        out["goal_error"] = str(exc)[:240]
    out["goal_wall_s"] = round(time.monotonic() - t, 2)

    # ---- criterion (iii): the planner, at the SHIPPED budget -------------------------------
    t = time.monotonic()
    diag: dict = {}
    try:
        plan = e3.plan_in_model(engine, goal if callable(goal) else None, root, diagnostics=diag)
        out["plan_found"] = bool(plan)
        out["plan_length"] = len(plan) if plan else 0
    except Exception as exc:  # noqa: BLE001
        out["plan_found"] = False
        out["plan_length"] = 0
        out["plan_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    out["plan_diagnostics"] = {
        k: v for k, v in diag.items() if isinstance(v, (int, float, str, bool))
    }
    out["plan_wall_s"] = round(time.monotonic() - t, 2)

    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
