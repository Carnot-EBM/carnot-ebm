#!/usr/bin/env python3
"""PHASE 2 analysis -- turn `phase2_raw.json` into the two answers, and the artifact.

NO GENERATED CODE RUNS HERE. Everything this file touches is either a recorded worker result or
a captured transition tape; the induced engines were executed only inside `worker.py`'s
subprocesses. That is what makes it safe to do the tape bookkeeping in-process.

(2A) WHY THE STALL PATH HAS NEVER CLEARED THE GOAL GATE. The shipped gate reports a single
undifferentiated `goal_unreached_within_depth` for three situations that call for three
completely different responses. The greedy rollout separates them:

  * INERT -- the engine predicts no action changes anything, from the root. Nothing is
    reachable, so no goal could be satisfied. The fault is in the DYNAMICS, not the goal.
  * DISPROVED -- the engine runs to its own dynamical fixed point (a state from which it
    predicts no further change) with the goal still false. The goal is genuinely unsatisfiable
    UNDER THIS ENGINE. This is strictly stronger than the gate's verdict: the gate says "I did
    not find it in 40 steps", the rollout says "the model cannot produce it at all".
  * BEYOND HORIZON -- the goal IS reached, at a depth past `max_depth=40`. The gate's negative
    was a search-horizon artifact and the predicate was right all along.

(2C) CAN A DYNAMICS ENGINE STEER EXPLORATION WITHOUT A CERTIFIED GOAL? The weakest useful
signal is "will this action change anything" -- an explorer that skips predicted no-ops spends
its action budget on actions that do something. Two things have to be true for that to be worth
anything, and they are measured separately because they turn out to disagree:

  1. HEADROOM -- a meaningful share of actions must actually be no-ops.
  2. INCREMENTAL VALUE OVER DEDUP -- the shipped explorer ALREADY avoids repeating a
     (state, action) pair it has tried. So the engine only adds something on FIRST-VISIT pairs,
     the ones dedup has no memory of. Scoring on all rows would credit the engine for
     rediscovering what dedup already knows -- the exact error the `repeat_penalty` work made
     when it improved a stage that was already at 1.0 in every live game.

  A degenerate engine that predicts "nothing ever changes" scores 100% on a tape that is 100%
  no-op, so accuracy is not reported alone: a game is gradeable for this question only if the
  held-out rows contain BOTH classes, and balanced accuracy is the headline.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import statistics

HERE = pathlib.Path(__file__).resolve().parent
SCRATCH = pathlib.Path("/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot") / (
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/phase2"
)
SHIPPED_MAX_DEPTH = 40  # `_goal_satisfiability_check(max_depth=40)`
SHIPPED_GATE_MAX_NODES = 20000  # `_GOAL_GATE_DEFAULT_MAX_NODES`
PRODUCTION_AFFORDABLE_ENGINE_CALLS = 17854  # recorded in arc_llm_reinduction.py


def classify_rollout(
    *,
    worker_status: str,
    rollout_status: str | None,
    depth_reached: int | None,
    goal_first_true_depth: int | None,
    shipped_max_depth: int = SHIPPED_MAX_DEPTH,
) -> str:
    """The taxonomy that separates the three causes the shipped gate reports as one verdict.

    The distinction that carries the phase is `inert_at_root` vs
    `disproved_at_engine_fixed_point`. Both surface from the worker as
    `engine_predicts_no_change_from_any_action`, but they mean opposite things: at depth 0 the
    engine never moved at all (a dynamics failure, the goal is irrelevant), while at depth > 0 it
    ran to a state from which it predicts no further change WITH THE GOAL STILL FALSE -- which
    disproves the goal under that engine. Collapsing them would merge 13 broken engines with 8
    genuine goal disproofs.
    """
    if worker_status != "ok":
        return f"not_scored:{worker_status.split(':')[0]}"
    if rollout_status == "goal_true_at_root_degenerate":
        return "degenerate_goal_true_at_root"
    if goal_first_true_depth is not None:
        return (
            "reachable_within_shipped_depth"
            if int(goal_first_true_depth) <= shipped_max_depth
            else "reachable_BEYOND_shipped_depth"
        )
    if rollout_status == "engine_predicts_no_change_from_any_action":
        return "inert_at_root" if int(depth_reached or 0) == 0 else "disproved_at_engine_fixed_point"
    if rollout_status == "max_steps_exhausted":
        return "cycling_no_goal_within_400_steps"
    return f"undetermined:{rollout_status}"


def balanced_accuracy(cm: dict) -> float | None:
    """Balanced accuracy, or None when a class is absent.

    None, never 0.0 or 1.0. A tape that is 100% no-op (lp85) is scored 1.0 by the degenerate
    engine that predicts nothing ever changes, and plain accuracy would report that as a
    triumph. With one class empty the quantity is UNDEFINED, and saying so is the only reading
    that cannot be mistaken for a result.
    """
    pos, neg = cm["tp"] + cm["fn"], cm["tn"] + cm["fp"]
    if pos == 0 or neg == 0:
        return None
    return round(0.5 * (cm["tp"] / pos + cm["tn"] / neg), 4)


def _pair(tr) -> tuple:
    d = tr.data if isinstance(tr.data, dict) else None
    return (int(tr.action), d.get("x") if d else None, d.get("y") if d else None)


def _skey(g) -> bytes | str:
    import numpy as np

    from carnot.agentic.arc_executable_world_model import _state_key

    k = _state_key(np.asarray(g))
    return k.hex() if isinstance(k, bytes) else str(k)


def tape_first_visit(game: str) -> dict[tuple, bool]:
    """Map (state_key, action, x, y) -> was this the FIRST time the tape stood there and did that.

    This is the information the explorer's own state-key dedup has. Anything the engine gets
    right on a REPEAT pair is something dedup already knew, so it cannot be counted as value
    added by the engine.
    """
    import numpy as np  # noqa: F401

    with open(SCRATCH / f"{game}_full.pkl", "rb") as fh:
        full = pickle.load(fh)
    seen: set[tuple] = set()
    out: dict[tuple, bool] = {}
    for tr in full:
        key = (_skey(tr.grid), *_pair(tr))
        out.setdefault(key, key not in seen)
        seen.add(key)
    return out


def main() -> int:  # noqa: C901
    import sys

    sys.path.insert(0, "/home/ianblenke/github.com/ianblenke/carnot/python")

    raw = json.loads((HERE.parent / "phase2_raw.json").read_text())
    results = raw["results"]
    games = sorted({r["game"] for r in results})
    stall = [g for g in games if g != "vc33"]  # vc33 is the POST-BANK path, reported apart

    firstvisit = {g: tape_first_visit(g) for g in games}

    # ================= (2A) rollout taxonomy ==============================================
    def classify(r: dict) -> str:
        ro = r.get("rollout") or {}
        return classify_rollout(
            worker_status=r.get("status", "unknown"),
            rollout_status=ro.get("status"),
            depth_reached=ro.get("depth_reached"),
            goal_first_true_depth=ro.get("goal_first_true_depth"),
        )

    taxonomy: dict[str, dict] = {}
    per_cand: list[dict] = []
    for r in results:
        cls = classify(r)
        ro = r.get("rollout") or {}
        rec = {
            "game": r["game"],
            "candidate": r["candidate"],
            "class": cls,
            "goal_depth": ro.get("goal_first_true_depth"),
            "depth_reached": ro.get("depth_reached"),
            "engine_calls": ro.get("engine_calls"),
            "n_distinct_states": ro.get("n_distinct_states"),
            "n_revisit_steps": ro.get("n_revisit_steps"),
            "rollout_wall_s": ro.get("wall_s"),
            "root_action_probe": r.get("root_action_probe"),
        }
        per_cand.append(rec)
        path = "postbank" if r["game"] == "vc33" else "stall"
        taxonomy.setdefault(path, {}).setdefault(cls, 0)
        taxonomy[path][cls] += 1

    beyond = [c for c in per_cand if c["class"] == "reachable_BEYOND_shipped_depth"]
    within = [c for c in per_cand if c["class"] == "reachable_within_shipped_depth"]

    # ================= (2C) change prediction, and its value over dedup ====================
    change_pred: list[dict] = []
    headroom: dict[str, dict] = {}
    for r in results:
        rows = r.get("heldout_rows")
        if not rows:
            continue
        g = r["game"]
        fv = firstvisit[g]
        n_chg = sum(1 for x in rows if x["actual_change"])
        headroom.setdefault(
            g,
            {
                "heldout_n": len(rows),
                "n_changing": n_chg,
                "n_noop": len(rows) - n_chg,
                "noop_fraction": round(1 - n_chg / len(rows), 4),
                "gradeable_both_classes": bool(0 < n_chg < len(rows)),
            },
        )
        cm = {"all": dict(tp=0, fp=0, tn=0, fn=0, unobserved=0), "first_visit": dict(tp=0, fp=0, tn=0, fn=0, unobserved=0)}
        for x in rows:
            key = (x["state_key"], x["action"], (x["data"] or {}).get("x"), (x["data"] or {}).get("y"))
            buckets = ["all"] + (["first_visit"] if fv.get(key, True) else [])
            for b in buckets:
                if x["pred_change"] is None:
                    cm[b]["unobserved"] += 1  # engine raised: a MISSING observation, not a "no"
                elif x["pred_change"] and x["actual_change"]:
                    cm[b]["tp"] += 1
                elif x["pred_change"] and not x["actual_change"]:
                    cm[b]["fp"] += 1
                elif not x["pred_change"] and not x["actual_change"]:
                    cm[b]["tn"] += 1
                else:
                    cm[b]["fn"] += 1

        bal_acc = balanced_accuracy
        change_pred.append(
            {
                "game": g,
                "candidate": r["candidate"],
                "confusion_all": cm["all"],
                "confusion_first_visit": cm["first_visit"],
                "balanced_accuracy_all": bal_acc(cm["all"]),
                "balanced_accuracy_first_visit": bal_acc(cm["first_visit"]),
                "n_exact": sum(1 for x in rows if x.get("pred_exact")),
            }
        )

    gradeable = [g for g, h in headroom.items() if h["gradeable_both_classes"]]
    ba = [c["balanced_accuracy_all"] for c in change_pred if c["balanced_accuracy_all"] is not None]
    ba_fv = [
        c["balanced_accuracy_first_visit"]
        for c in change_pred
        if c["balanced_accuracy_first_visit"] is not None
    ]

    out = {
        "shipped_max_depth": SHIPPED_MAX_DEPTH,
        "shipped_gate_max_nodes": SHIPPED_GATE_MAX_NODES,
        "production_affordable_engine_calls": PRODUCTION_AFFORDABLE_ENGINE_CALLS,
        "rollout_taxonomy": taxonomy,
        "per_candidate": sorted(per_cand, key=lambda x: (x["game"], x["candidate"])),
        "reachable_beyond_shipped_depth": [
            {k: c[k] for k in ("game", "candidate", "goal_depth", "engine_calls", "rollout_wall_s")}
            for c in beyond
        ],
        "reachable_within_shipped_depth": [
            {k: c[k] for k in ("game", "candidate", "goal_depth", "engine_calls", "rollout_wall_s")}
            for c in within
        ],
        "change_prediction": {
            "headroom_by_game": headroom,
            "gradeable_games": gradeable,
            "n_gradeable_games": len(gradeable),
            "per_candidate": change_pred,
            "balanced_accuracy_all_median": round(statistics.median(ba), 4) if ba else None,
            "balanced_accuracy_first_visit_median": round(statistics.median(ba_fv), 4) if ba_fv else None,
            "n_candidates_with_defined_balanced_accuracy": len(ba),
            "n_candidates_with_defined_first_visit_balanced_accuracy": len(ba_fv),
        },
        "stall_games": stall,
    }
    (HERE.parent / "phase2_analysis.json").write_text(json.dumps(out, indent=1, sort_keys=True))

    # ---- console summary --------------------------------------------------------------------
    print("=== (2A) rollout taxonomy ===")
    for path in sorted(taxonomy):
        print(f"  {path}:")
        for k, v in sorted(taxonomy[path].items(), key=lambda kv: -kv[1]):
            print(f"    {v:3d}  {k}")
    print(f"\n  reachable BEYOND shipped depth {SHIPPED_MAX_DEPTH}: {len(beyond)}")
    for c in beyond:
        print(f"    {c['game']} k{c['candidate']}: depth {c['goal_depth']}, {c['engine_calls']} engine calls, {c['rollout_wall_s']}s")
    print("\n=== (2C) change prediction ===")
    for g, h in sorted(headroom.items()):
        print(f"  {g}: noop_frac={h['noop_fraction']:.2f} gradeable={h['gradeable_both_classes']}")
    print(f"  gradeable games (both classes present): {gradeable}")
    print(f"  balanced acc  all rows median: {out['change_prediction']['balanced_accuracy_all_median']} (n={len(ba)})")
    print(f"  balanced acc  FIRST-VISIT median: {out['change_prediction']['balanced_accuracy_first_visit_median']} (n={len(ba_fv)})")
    print(f"\nwrote {HERE.parent / 'phase2_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
