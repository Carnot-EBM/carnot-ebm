#!/usr/bin/env python3
"""Score the best-of-N candidates for ONE game on that corpus's PROVEN held-out split.

WHY THIS CORPUS GETS ITS OWN WORKER. Its held-out set is not a fraction rule -- it is
`full \\ shown` where `shown` is the rows the prompt ACTUALLY RENDERED, proven row-by-row
against the prompt TEXT, with ambiguous rows DROPPED rather than counted as unseen. That is
the strongest never-fitted guarantee in this repo, and it is the reason this corpus is worth
a second code path instead of being forced through the window-tail splitter.

Two pins, both copied from results/arc_metric_validity_20260801/bon_window_worker.py, whose
docstring explains them: SPLIT_CALL_INDEX=1 (bestofn_scored.json records call_index 1; the
module default is 2) and CARNOT_ARC_INDUCE_TRANSITIONS_K=8 (the live resolver's default became
None on 2026-08-01, the day AFTER this corpus was frozen -- unpinned, the split silently
reshapes and every held-out number grades a different row set). The rebuilt split is then
CHECKED against the frozen split.json row counts; a mismatch VOIDS the game rather than being
averaged in.
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
os.environ["CARNOT_ARC_E3_DIR"] = os.environ["SCRATCH_E3"]
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
os.environ["SPLIT_CALL_INDEX"] = "1"
os.environ["CARNOT_ARC_INDUCE_TRANSITIONS_K"] = "8"
sys.path.insert(0, str(REPO / "python"))

ENGINE_TIMEOUT_S = int(os.environ.get("GVS_ENGINE_TIMEOUT_S", "90"))


class _TimeoutError(Exception):
    pass


def _alarm(_s, _f):
    raise _TimeoutError()


def main() -> int:
    game = sys.argv[1]
    jobs = json.loads(pathlib.Path(sys.argv[2]).read_text())
    outpath = pathlib.Path(sys.argv[3])
    bestofn = REPO / "results" / "arc_induce_bestofn_20260731"
    sys.path.insert(0, str(bestofn / "harness"))

    import numpy as np

    from split import load_split  # type: ignore[import-not-found]

    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_world_model_trust_energy as wmte

    out: dict = {
        "game": game,
        "corpus": "bestofn",
        "hidden_state_branch": game in wmte.HIDDEN_STATE_GAME_IDS,
    }

    s = load_split(game, 1)
    shown, held = list(s["_shown"]), list(s["_heldout"])
    frozen = {r["game"]: r for r in json.loads((bestofn / "split.json").read_text())["rows"]}.get(
        game
    )
    out["split_check"] = {
        "n_shown_rebuilt": len(shown),
        "n_heldout_rebuilt": len(held),
        "n_shown_frozen": (frozen or {}).get("n_shown"),
        "n_heldout_frozen": (frozen or {}).get("n_heldout"),
        "split_proven_frozen": (frozen or {}).get("split_proven"),
        "heldout_n_noop_frozen": (frozen or {}).get("heldout_n_noop"),
    }
    if not frozen or len(shown) != frozen["n_shown"] or len(held) != frozen["n_heldout"]:
        out["status"] = "split_mismatch_game_voided"
        outpath.write_text(json.dumps(out, default=str))
        print(f"GVS_BON_VOID {game}", flush=True)
        return 0
    out["status"] = "ok"

    def describe(ts):
        chg = [t for t in ts if not np.array_equal(np.asarray(t.grid), np.asarray(t.next_grid))]
        grad = [
            t for t in chg if not (getattr(t, "level_after", 0) > getattr(t, "level_before", 0))
        ]
        return {"n": len(ts), "n_changing": len(chg), "n_gradable_changing": len(grad)}

    splits = {"P_proven_heldout": held}
    try:
        fresh, _c = e3.collect_transitions(game, n=120, seed=20260802)
        splits["C_fresh120"] = list(fresh)
    except Exception as exc:
        out["fresh_error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    out["splits"] = {k: describe(v) for k, v in splits.items()}

    def score_one(engine, ts) -> dict:
        vr = e3.WorldModelVerifier(list(ts)).score(engine)
        cw = wmte.score_change_weighted_consistency(list(ts), engine)
        gate = e3.change_gate_decision(vr, enabled=True)  # else passed==True/gate_disabled
        c, sp = int(vr.correct_changed_cells), int(vr.spurious_changed_cells)
        return {
            "n_scored": int(vr.n),
            "vr_n_changing": int(vr.n_changing),
            "noop_hallucination_rate": round(float(vr.noop_hallucination_rate), 6),
            "accuracy": round(float(vr.accuracy), 6),
            "plain_exact_pass_0p5": bool(float(vr.accuracy) >= 0.5),
            "refinement_loop_pass_1p0": bool(float(vr.accuracy) >= 1.0),
            "cell_recall": round(float(vr.cell_recall), 6),
            "plain_cellrecall_pass_0p5": bool(float(vr.cell_recall) >= 0.5),
            "change_accuracy": round(float(vr.change_accuracy), 6),
            "heldout_change_consistency": round(float(cw.consistency), 6),
            "cw_correct_changed_cells": int(cw.correct_changed_cells),
            "cw_true_changed_cells": int(cw.true_changed_cells),
            "cw_nondegenerate": bool(cw.nondegenerate),
            "hidden_state_trust_pass": bool(cw.trust_pass),
            "change_fidelity": round(float(vr.change_fidelity), 6),
            "change_gate_pass": bool(gate.get("passed")),
            "change_gate_reason": str(gate.get("reason")),
            "correct_changed_cells": c,
            "spurious_changed_cells": sp,
            "precision": round(c / (c + sp), 6) if (c + sp) > 0 else None,
        }

    def identity(grid, action, data):
        return grid

    rows = []
    for sname, ts in splits.items():
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
                        "split": sname,
                        "status": "ok",
                        **score_one(fn, ts),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "cell": cname,
                        "corpus": "control",
                        "split": sname,
                        "status": f"error:{type(exc).__name__}",
                    }
                )

    for job in jobs:
        p = pathlib.Path(job["path"])
        base = {
            "cell": job["cell"],
            "corpus": "bestofn",
            "engine_path": str(p),
            "engine_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        try:
            spec = importlib.util.spec_from_file_location(f"gvsb_{p.stem}_{os.getpid()}", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            engine = getattr(mod, "engine", None)
            base["has_goal_predicate"] = getattr(mod, "is_level_complete", None) is not None
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
        for sname, ts in splits.items():
            if not ts:
                continue
            t0 = time.time()
            signal.signal(signal.SIGALRM, _alarm)
            signal.alarm(ENGINE_TIMEOUT_S)
            try:
                rows.append({**base, "split": sname, "status": "ok", **score_one(engine, ts)})
                rows[-1]["wall_s"] = round(time.time() - t0, 3)
            except _TimeoutError:
                rows.append({**base, "split": sname, "status": "engine_timeout"})
            except Exception as exc:
                rows.append(
                    {
                        **base,
                        "split": sname,
                        "status": f"score_error:{type(exc).__name__}",
                        "detail": str(exc)[:160],
                    }
                )
            finally:
                signal.alarm(0)

    out["rows"] = rows
    outpath.write_text(json.dumps(out, default=str))
    print(f"GVS_BON_DONE {game} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
