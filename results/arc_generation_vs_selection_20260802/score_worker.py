#!/usr/bin/env python3
"""Score every saved engine for ONE game on three never-fitted splits, in a killable process.

WHY ONE PROCESS PER GAME. Two independent hazards, both already observed in this repo:
`build_progress_window` has no internal bound (tr87 span at 100% CPU and took two
separately-written drivers down), and the engines are LLM-WRITTEN CODE that can loop or
allocate without limit. The driver applies the outer timeout; SIGALRM applies a per-engine
one inside. Anything that trips either is recorded with a STATUS and excluded -- never
scored 0. Missing is not zero.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import signal
import sys
import time

REPO = pathlib.Path(os.environ["CARNOT_REPO"])
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# results/arc_e3 is EVIDENCE. Point every writer at scratch BEFORE the import that reads it.
os.environ["CARNOT_ARC_E3_DIR"] = os.environ["SCRATCH_E3"]
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, str(REPO / "python"))

ENGINE_TIMEOUT_S = int(os.environ.get("GVS_ENGINE_TIMEOUT_S", "90"))


class _TimeoutError(Exception):
    pass


def _alarm(_s, _f):
    raise _TimeoutError()


def load_engine_from_file(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(f"gvs_{path.stem}_{os.getpid()}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, "engine", None), getattr(mod, "is_level_complete", None)


def main() -> int:
    game = sys.argv[1]
    jobs = json.loads(pathlib.Path(sys.argv[2]).read_text())

    import numpy as np

    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_world_model_trust_energy as wmte

    out: dict = {"game": game, "hidden_state_branch": game in wmte.HIDDEN_STATE_GAME_IDS}

    # ---------------- splits ----------------
    splits: dict[str, list] = {}
    try:
        w = atp.build_progress_window(game)
    except Exception as exc:
        w = None
        out["window_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    if w is not None:
        win, full, cell = w
        shown, held = wmte._split_prefix_heldout(list(win))  # noqa: SLF001
        out["cell"] = int(cell)
        out["n_window"], out["n_full"], out["n_shown"] = len(win), len(full), len(shown)

        def sig(t):
            return hashlib.sha256(
                np.ascontiguousarray(np.asarray(t.grid)).tobytes()
                + b"|"
                + str(getattr(t, "action", None)).encode()
                + b"|"
                + np.ascontiguousarray(np.asarray(t.next_grid)).tobytes()
            ).hexdigest()

        shown_sigs = {sig(t) for t in shown}
        splits["A_tail"] = list(held)
        splits["B_rest"] = [t for t in full if sig(t) not in shown_sigs]
    try:
        fresh, _fcell = e3.collect_transitions(game, n=120, seed=20260802)
        splits["C_fresh120"] = list(fresh)
    except Exception as exc:
        out["fresh_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"

    def describe(ts):
        chg = [t for t in ts if not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))]
        grad = [
            t for t in chg if not (getattr(t, "level_after", 0) > getattr(t, "level_before", 0))
        ]
        return {"n": len(ts), "n_changing": len(chg), "n_gradable_changing": len(grad)}

    out["splits"] = {k: describe(v) for k, v in splits.items()}

    # ---------------- metrics ----------------
    def score_one(engine, ts) -> dict:
        """Every SHIPPED gate quantity, on one split, for one engine."""
        vr = e3.WorldModelVerifier(list(ts)).score(engine)
        cw = wmte.score_change_weighted_consistency(list(ts), engine)  # the hidden-state gate
        gate = e3.change_gate_decision(vr, enabled=True)  # else passed==True/gate_disabled
        c = int(vr.correct_changed_cells)
        s = int(vr.spurious_changed_cells)
        return {
            "n_scored": int(vr.n),
            "vr_n_changing": int(vr.n_changing),
            "noop_hallucination_rate": round(float(vr.noop_hallucination_rate), 6),
            # plain branch, default metric + its live thresholds
            "accuracy": round(float(vr.accuracy), 6),
            "plain_exact_pass_0p5": bool(float(vr.accuracy) >= 0.5),
            "refinement_loop_pass_1p0": bool(float(vr.accuracy) >= 1.0),
            # plain branch under the already-shipped default-OFF cell_recall flag
            "cell_recall": round(float(vr.cell_recall), 6),
            "plain_cellrecall_pass_0p5": bool(float(vr.cell_recall) >= 0.5),
            # the de-inflated exact number: whole grid right on CHANGING rows only
            "change_accuracy": round(float(vr.change_accuracy), 6),
            # the hidden-state branch gate, POOLED over changed cells (not the mean cell_recall)
            "heldout_change_consistency": round(float(cw.consistency), 6),
            "cw_correct_changed_cells": int(cw.correct_changed_cells),
            "cw_true_changed_cells": int(cw.true_changed_cells),
            "cw_nondegenerate": bool(cw.nondegenerate),
            "hidden_state_trust_pass": bool(cw.trust_pass),
            # the default-OFF symmetric gate
            "change_fidelity": round(float(vr.change_fidelity), 6),
            "change_gate_pass": bool(gate.get("passed")),
            "change_gate_reason": str(gate.get("reason")),
            # precision: a high recall with low precision is an engine that SCRIBBLES
            "correct_changed_cells": c,
            "spurious_changed_cells": s,
            "precision": round(c / (c + s), 6) if (c + s) > 0 else None,
        }

    def identity(grid, action, data):
        return grid

    rows = []
    # oracle is built per-split: it must see the rows it is graded on, which is exactly
    # why it is an INSTRUMENT CHECK and not a result.
    for split_name, ts in splits.items():
        if not ts:
            continue
        # THE STATELESS CEILING, not merely a plumbing check. Engines are functions of
        # (grid, action, data) with no state carried between calls, so the best ANY of them
        # can do is the BAYES-OPTIMAL stateless predictor: for each (grid, action) return the
        # MODAL next_grid. A last-write-wins lookup is NOT that -- where one (grid, action)
        # leads to different successors (hidden state) it keeps whichever came last and can
        # score BELOW a real engine, which is how the first version of this control was
        # caught: real lp85 engines scored 0.717 against a 0.25 "oracle".
        from collections import Counter

        counts: dict = {}
        shapes: dict = {}
        for t in ts:
            k = (np.asarray(t.grid).tobytes(), str(getattr(t, "action", None)))
            nb = np.ascontiguousarray(np.asarray(t.next_grid))
            counts.setdefault(k, Counter())[nb.tobytes()] += 1
            shapes[(k, nb.tobytes())] = (nb.shape, nb.dtype)

        def oracle(grid, action, data, _c=counts, _s=shapes):
            k = (np.asarray(grid).tobytes(), str(action))
            c = _c.get(k)
            if not c:
                return grid
            best = c.most_common(1)[0][0]
            shp, dt = _s[(k, best)]
            return np.frombuffer(best, dtype=dt).reshape(shp)

        for cname, fn in (("__control_oracle__", oracle), ("__control_identity__", identity)):
            try:
                rows.append(
                    {
                        "cell": cname,
                        "corpus": "control",
                        "split": split_name,
                        "status": "ok",
                        **score_one(fn, ts),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "cell": cname,
                        "corpus": "control",
                        "split": split_name,
                        "status": f"error:{type(exc).__name__}",
                    }
                )

    for job in jobs:
        p = pathlib.Path(job["path"])
        base = {
            "cell": job["cell"],
            "corpus": job["corpus"],
            "engine_path": str(p),
            "engine_sha256": job["sha256"],
        }
        try:
            engine, is_done = load_engine_from_file(p)
        except Exception as exc:
            rows.append(
                {
                    **base,
                    "split": None,
                    "status": f"unloadable:{type(exc).__name__}",
                    "detail": str(exc)[:160],
                }
            )
            continue
        if engine is None:
            rows.append({**base, "split": None, "status": "no_engine_attribute"})
            continue
        base["has_goal_predicate"] = is_done is not None
        for split_name, ts in splits.items():
            if not ts:
                continue
            if job["corpus"] == "e3store" and split_name in ("A_tail", "B_rest"):
                continue  # induction window unknown -> A/B undefined; prereg says C only
            t0 = time.time()
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(ENGINE_TIMEOUT_S)
            try:
                rows.append(
                    {
                        **base,
                        "split": split_name,
                        "status": "ok",
                        "wall_s": None,
                        **score_one(engine, ts),
                    }
                )
                rows[-1]["wall_s"] = round(time.time() - t0, 3)
            except _TimeoutError:
                rows.append(
                    {
                        **base,
                        "split": split_name,
                        "status": "engine_timeout",
                        "timeout_s": ENGINE_TIMEOUT_S,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        **base,
                        "split": split_name,
                        "status": f"score_error:{type(exc).__name__}",
                        "detail": str(exc)[:160],
                    }
                )
            finally:
                signal.alarm(0)

    out["rows"] = rows
    pathlib.Path(sys.argv[3]).write_text(json.dumps(out, default=str))
    print(f"GVS_DONE {game} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
