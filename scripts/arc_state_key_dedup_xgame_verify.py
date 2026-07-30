"""PHASE 1(b): cross-game verification of the bytes dedup key, before shipping it globally.

WHAT IS BEING DECIDED. `plan_in_model`'s duplicate-state test currently keys on `to_ascii(ng)` --
a one-char-per-cell Python string built by a nested `str.join` loop, which the cProfile of a
shipped-budget search measured at 38% of total search time. Replacing it with a NumPy `tobytes()`
key was measured 1.288x faster on ka59 for the SAME 137,347 engine calls. This script asks whether
that result generalises before the swap is made everywhere.

WHY A `% 10` STEP IS NOT OPTIONAL, and why a plain `tobytes()` is NOT the drop-in.
`to_ascii` renders each cell as `str(int(v))[-1]` -- the LAST DIGIT ONLY. So colour 14 and colour 4
render to the same character, as do 15/5, 11/1, 10/0. The shipped key is therefore LOSSY: it MERGES
two states that differ only by such a swap, and the search discards the second as already-seen.
Those colours are not hypothetical -- ka59's root grid contains both 4 and 14 AND both 5 and 15,
and lp85's contains 1/11, 4/14 and 5/15. A plain `tobytes()` distinguishes them, so it induces a
STRICTLY FINER partition: a semantic change to the search, which may explore more states and find a
different plan. `(g % 10).astype(uint8).tobytes()` instead reproduces `to_ascii`'s equivalence
classes EXACTLY for every NON-NEGATIVE integer, which is what makes it a drop-in rather than a
change.

NOTE the non-negative qualifier -- an earlier draft of this file claimed "for every integer,
including negatives" and that was WRONG. `to_ascii` takes the last character of the DECIMAL STRING,
so for a negative number it is the last digit of the ABSOLUTE value (`str(-1)[-1] == '1'`), whereas
`-1 % 10 == 9`. They agree only where a digit is its own complement mod 10 (0 and 5), so they
disagree on 12 of the 16 values in -15..-1. The landed `_state_key` therefore guards its fast path on
`a.min() >= 0` and defers everything else to `to_ascii` itself. The `mod10_bytes` arm below is the
UNGUARDED idea; the `shipped_state_key` arm is the landed function including that guard, and it is
the arm whose verdict decides whether the swap is safe.

Both are measured here, and reported separately, so the speed win is never smuggled in alongside a
partition change.

HOW THE PARTITION IS PROVED IDENTICAL -- not by counts.
Equal `engine_calls` and equal `unique_states` are necessary but weak: two different partitions can
coincidentally agree on both totals. The strong instrument is the ACCEPT TRACE: for every engine
call, in order, one byte recording what the search DECIDED -- 'A' accepted as a new state, 'D'
rejected as a duplicate, 'S' skipped on a shape mismatch, 'E' the engine raised. That sequence is
key-INDEPENDENT (it records decisions, not keys), it is the complete record of how the dedup key
influenced the search, and any divergence anywhere shows up as a different hash. If two arms share
an accept-trace hash and a plan, they ran the identical search.

BUDGET. Every arm runs to the SAME engine-call cap, so the wall-clock comparison is like-for-like
(equal work), and when the cap is reached rather than a plan found, the partition comparison covers
the entire explored prefix rather than stopping at the first goal state.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import deque

import numpy as np

REPO = os.environ.get("CARNOT_REPO_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
# Output goes to $XG_OUT_DIR, else the CURRENT DIRECTORY -- deliberately NOT next to this file.
# When this harness lived in a scratch dir, "next to the script" was the right default; once it moved
# into `scripts/` that same line started dropping `xgame_dedup_cap*.json` into a tracked source
# directory, which a later `git add -A` would have committed as stray output. Caught by a post-commit
# smoke run showing an untracked file under `scripts/`.
OUT_DIR = os.environ.get("XG_OUT_DIR") or os.getcwd()
# The engine store is redirected by default: `results/arc_e3` is TRACKED, read-only evidence, and an
# induction triggered from here must never write into it.
os.environ.setdefault(
    "CARNOT_ARC_E3_DIR", os.path.join(tempfile.gettempdir(), "arc_state_key_verify_e3")
)
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(REPO, "python"))

from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_agi3_world_model import grid_of  # noqa: E402
from carnot.agentic.arc_executable_world_model import (  # noqa: E402
    _model_candidates,
    _state_key,
    detect_cell,
    to_ascii,
    to_logical,
)

CAP = int(os.environ.get("XG_CAP", "20000"))
MAX_DEPTH = 40
GAMES = os.environ.get("XG_GAMES", "ka59,lp85,tr87,vc33,cn04,sk48").split(",")


def key_ascii(a):
    """The SHIPPED key."""
    return to_ascii(a)


def key_mod10(a):
    """The DROP-IN: identical equivalence classes to `to_ascii`, computed in C."""
    return (np.asarray(a) % 10).astype(np.uint8).tobytes()


def key_exact(a):
    """STRICTLY FINER than shipped -- a semantic change, reported, never shipped as a speedup."""
    return np.ascontiguousarray(a).tobytes()


def key_shipped(a):
    """The ACTUAL landed `_state_key`, imported from the module.

    `key_mod10` above is a local reimplementation of the idea. This arm is the code that really
    runs, including its shape prefix and its non-integer-dtype fallback, so the verification
    proves the SHIPPED function equivalent rather than a stand-in for it.
    """
    return _state_key(a)


def bfs(engine, goal, root, keyfn, cap):
    """Blind FIFO BFS, structurally identical to `plan_in_model`'s `goal_energy=None` branch.

    Returns the plan, the counters, and the ACCEPT TRACE hash described in the module docstring.
    """
    seen = {keyfn(root)}
    q = deque([(root, [])])
    calls = 0
    trace = bytearray()
    while q and calls < cap:
        grid, path = q.popleft()
        if len(path) >= MAX_DEPTH:
            continue
        for c in _model_candidates(grid):
            try:
                ng = np.asarray(engine(grid.copy(), c["action"], c["data"]))
            except Exception:
                calls += 1
                trace += b"E"
                if calls >= cap:
                    break
                continue
            calls += 1
            if ng.shape != root.shape:
                trace += b"S"
                if calls >= cap:
                    break
                continue
            k = keyfn(ng)
            if k in seen:
                trace += b"D"
                if calls >= cap:
                    break
                continue
            seen.add(k)
            trace += b"A"
            npath = path + [c]
            try:
                if bool(goal(ng)):
                    return (
                        npath,
                        calls,
                        len(seen),
                        "plan_found",
                        hashlib.sha256(trace).hexdigest(),
                        len(trace),
                    )
            except Exception:
                pass
            q.append((ng, npath))
            if calls >= cap:
                break
    return (
        None,
        calls,
        len(seen),
        ("cap_reached" if calls >= cap else "queue_exhausted"),
        hashlib.sha256(trace).hexdigest(),
        len(trace),
    )


# A game name may be written `<game>@<git-rev>` to pin a HISTORICAL engine instead of the one
# currently on disk. This exists because several games' CURRENT on-disk engines exhaust their
# search queue in under 100 calls, which yields a correct partition comparison but no usable
# timing signal at all -- and because the 137,347-call ka59 measurement the whole lever rests on
# was made against `341f776c9`'s engine, not today's 25-line one, so citing today's engine as
# "ka59" would silently compare against a different program.
def load_engine(game):
    rel = f"results/arc_e3/{game.split('@')[0]}/world_model.py"
    if "@" in game:
        rev = game.split("@", 1)[1]
        src = subprocess.run(
            ["git", "-C", REPO, "show", f"{rev}:{rel}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    else:
        src = open(os.path.join(REPO, rel)).read()
    ns: dict = {}
    exec(compile(src, f"<e3_{game}>", "exec"), ns)
    return ns.get("engine"), ns.get("is_level_complete"), hashlib.sha256(src.encode()).hexdigest()


def main():
    arc = kit.offline_arcade()
    sc = arc.open_scorecard()
    out = {
        "experiment": "p3_xgame_dedup_key_verification",
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "is_a_solve_claim": False,
        "engine_call_cap_per_arm": CAP,
        "max_depth": MAX_DEPTH,
        "games": {},
    }
    for game in GAMES:
        rec: dict = {"game": game}
        try:
            engine, goal, esha = load_engine(game)
            if engine is None:
                rec["status"] = "no_engine_symbol"
                out["games"][game] = rec
                print(json.dumps(rec), flush=True)
                continue
            rec["engine_sha256"] = esha
            env = arc.make(game.split("@")[0], scorecard_id=sc)
            frame = env.reset()
            raw = grid_of(frame)
            cell = detect_cell(raw)
            root = np.asarray(to_logical(raw, cell))
            rec["logical_shape"] = list(root.shape)
            rec["logical_dtype"] = str(root.dtype)
            cols = sorted(int(v) for v in np.unique(root).tolist())
            rec["root_colours"] = cols
            # The pairs the SHIPPED key cannot tell apart, present in this very grid.
            rec["aliasing_pairs_present_in_root"] = [
                [a, b] for a in cols for b in cols if a < b and a % 10 == b % 10
            ]
            if goal is None:
                # No goal predicate: the search still runs to the cap, which is what the
                # partition comparison needs. Recorded so the row is not misread as a solve test.
                def goal(_g):
                    return False

                rec["goal_predicate_present"] = False
            else:
                rec["goal_predicate_present"] = True

            # WARM-UP, and it is not optional. The first arm to run pays one-off costs the others
            # do not: the engine's module-level lazies, NumPy's first-touch allocations, the
            # candidate generator's caches. In the first version of this script the control arm ran
            # first and absorbed 0.74s of that on ka59, which the ratio then reported as a "371x
            # speedup" -- an artefact of arm ORDER, not of the key. Every arm is now preceded by a
            # discarded short run, and each arm is timed TWICE with the MINIMUM taken, because the
            # minimum of repeated timings is the estimator least contaminated by one-off costs and
            # by scheduler noise.
            bfs(engine, goal, root, key_ascii, min(400, CAP))
            arms = {}
            for name, fn in (
                ("to_ascii_control", key_ascii),
                ("mod10_bytes", key_mod10),
                ("exact_bytes", key_exact),
                ("shipped_state_key", key_shipped),
            ):
                best = None
                for _rep in range(2):
                    t0 = time.perf_counter()
                    plan, calls, uniq, reason, thash, tlen = bfs(engine, goal, root, fn, CAP)
                    el = time.perf_counter() - t0
                    best = el if best is None else min(best, el)
                dt = best
                arms[name] = {
                    "plan_found": plan is not None,
                    "plan_length": len(plan) if plan else None,
                    "plan": [{"action": p["action"], "data": p["data"]} for p in (plan or [])]
                    or None,
                    "engine_calls": calls,
                    "unique_states": uniq,
                    "termination_reason": reason,
                    "accept_trace_sha256": thash,
                    "accept_trace_len": tlen,
                    "wall_clock_s": round(dt, 3),
                    "engine_calls_per_s": round(calls / dt, 1) if dt > 0 else None,
                }
                print(
                    f"  {game:6s} {name:17s} calls={calls} uniq={uniq} "
                    f"{reason} {dt:.2f}s trace={thash[:12]}",
                    flush=True,
                )
            rec["arms"] = arms
            ctl, m10, ex = arms["to_ascii_control"], arms["mod10_bytes"], arms["exact_bytes"]
            sh = arms["shipped_state_key"]

            def same(a, b):
                return {
                    "same_engine_calls": a["engine_calls"] == b["engine_calls"],
                    "same_unique_states": a["unique_states"] == b["unique_states"],
                    "same_plan": a["plan"] == b["plan"],
                    "same_accept_trace_sha256": a["accept_trace_sha256"]
                    == b["accept_trace_sha256"],
                }

            m = same(ctl, m10)
            rec["PARTITION_IDENTICAL_mod10_vs_shipped"] = dict(
                m,
                verdict=(
                    "PARTITION_IDENTICAL" if all(m.values()) else "PARTITION_DIFFERS_DO_NOT_SHIP"
                ),
            )
            e = same(ctl, ex)
            rec["PARTITION_IDENTICAL_exact_vs_shipped"] = dict(
                e,
                verdict=(
                    "coincidentally_identical_on_this_game"
                    if all(e.values())
                    else "SEMANTIC_CHANGE_as_predicted"
                ),
            )
            # A ratio between two sub-10ms runs is rounding noise, and a queue that exhausted in
            # a few dozen calls never exercised the dedup key at all. Both are recorded as
            # partition evidence and EXCLUDED from the speed table rather than quietly averaged in.
            rec["timing_usable"] = bool(
                ctl["termination_reason"] == "cap_reached" and ctl["wall_clock_s"] >= 0.5
            )
            rec["timing_unusable_reason"] = (
                None
                if rec["timing_usable"]
                else (
                    f"control ended {ctl['termination_reason']} after {ctl['engine_calls']} calls "
                    f"in {ctl['wall_clock_s']}s -- too little work to time"
                )
            )
            msh = same(ctl, sh)
            rec["PARTITION_IDENTICAL_landed_state_key_vs_shipped_to_ascii"] = dict(
                msh,
                verdict=(
                    "PARTITION_IDENTICAL" if all(msh.values()) else "PARTITION_DIFFERS_DO_NOT_SHIP"
                ),
            )
            rec["speedup_landed_state_key_over_to_ascii"] = (
                round(ctl["wall_clock_s"] / max(1e-9, sh["wall_clock_s"]), 4)
                if ctl["termination_reason"] == "cap_reached" and ctl["wall_clock_s"] >= 0.5
                else None
            )
            rec["speedup_mod10_over_shipped"] = (
                round(ctl["wall_clock_s"] / max(1e-9, m10["wall_clock_s"]), 4)
                if rec["timing_usable"]
                else None
            )
            rec["speedup_exact_over_shipped"] = (
                round(ctl["wall_clock_s"] / max(1e-9, ex["wall_clock_s"]), 4)
                if rec["timing_usable"]
                else None
            )
            rec["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            rec["status"] = f"error:{type(exc).__name__}"
            rec["error"] = str(exc)[:400]
        out["games"][game] = rec
        print(json.dumps({k: v for k, v in rec.items() if k != "arms"}, default=str), flush=True)

    ok = [r for r in out["games"].values() if r.get("status") == "ok"]
    out["SUMMARY"] = {
        "n_games_measured": len(ok),
        "all_mod10_partitions_identical": bool(ok)
        and all(
            r["PARTITION_IDENTICAL_mod10_vs_shipped"]["verdict"] == "PARTITION_IDENTICAL"
            for r in ok
        ),
        "n_games_with_usable_timing": sum(1 for r in ok if r.get("timing_usable")),
        "mod10_speedup_by_game": {
            r["game"]: r["speedup_mod10_over_shipped"] for r in ok if r.get("timing_usable")
        },
        "mod10_min_speedup": min(
            (r["speedup_mod10_over_shipped"] for r in ok if r.get("timing_usable")), default=None
        ),
        "partition_verified_but_timing_unusable": {
            r["game"]: r["timing_unusable_reason"] for r in ok if not r.get("timing_usable")
        },
        "all_landed_state_key_partitions_identical": bool(ok)
        and all(
            r["PARTITION_IDENTICAL_landed_state_key_vs_shipped_to_ascii"]["verdict"]
            == "PARTITION_IDENTICAL"
            for r in ok
        ),
        "landed_state_key_speedup_by_game": {
            r["game"]: r["speedup_landed_state_key_over_to_ascii"]
            for r in ok
            if r.get("timing_usable")
        },
        "games_where_exact_key_differs": [
            r["game"]
            for r in ok
            if r["PARTITION_IDENTICAL_exact_vs_shipped"]["verdict"]
            != ("coincidentally_identical_on_this_game")
        ],
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, f"xgame_dedup_cap{CAP}.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(json.dumps(out["SUMMARY"], indent=2, default=str))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
