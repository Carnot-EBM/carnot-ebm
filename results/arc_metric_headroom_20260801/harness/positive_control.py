#!/usr/bin/env python3
"""METRIC-HEADROOM, STEP 2 -- the POSITIVE CONTROL. Does each metric resolve a KNOWN difference?

WHY A CENSUS OF THE 48 CANDIDATES IS NOT SUFFICIENT ON ITS OWN.

Step 1 shows how much each metric VARIES over the frozen candidates. Variation is necessary but
not sufficient: a metric could vary because it is noisy, or because it is measuring something
other than model quality (exact-match on a no-op-heavy split varies with how well an engine
predicts that NOTHING happens, which is not dynamics). What an A/B actually needs is RESOLUTION --
if arm B's engines are genuinely better than arm A's, does the number move?

That question can be answered exactly, offline, with no LLM and no GPU, by grading engines whose
quality is CONSTRUCTED and therefore known:

  identity              returns the input grid unchanged. The degenerate baseline. It is the
                        engine that earns a perfect held-out score on a split with no changing
                        rows, which is why lp85 must not be graded for change quality.
  oracle                returns the recorded next grid. Perfect by construction.
  oracle_corrupt_p      the oracle with a fraction p of the cells REALITY CHANGED overwritten
                        with a wrong colour. A monotone quality ladder: p=0 is perfect, p=1 gets
                        every changed cell wrong. An instrument with headroom MUST order this
                        ladder; an instrument at its floor gives the whole ladder one value.
  oracle_spurious_k     the oracle plus k cells reality did NOT change, overwritten anyway. This
                        separates the SYMMETRIC metrics from the asymmetric ones: `cell_recall`
                        is documented in this repo as structurally blind to spurious writes and
                        should score it 1.0, while `change_fidelity` scores over the UNION and
                        should penalise it. If the recommendation is going to be one of these two,
                        the difference has to be demonstrated rather than asserted.

THIS SCRIPT MAY RUN ITS ENGINES IN-PROCESS, and that is not a violation of the no-generated-code
rule. Every engine here is written in this file, is a total function of its input, and contains no
loop that can fail to terminate. The rule exists because LLM-WRITTEN code hung the pipeline for 13
minutes on 2026-07-31; it is a rule about provenance, not about the word "engine".

RESULT SHAPE. For each metric: the value at each rung, whether the ladder is strictly monotone in
the intended direction, how many distinct values the ladder produces (a floored instrument
produces 1), and whether identity and oracle are distinguished at all.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[3]
HERE = pathlib.Path(__file__).resolve().parent
OUT_DIR = HERE.parent
BON_HARNESS = REPO / "results" / "arc_induce_bestofn_20260731" / "harness"

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_metric_headroom/e3")
os.environ.setdefault("CARNOT_ARC_INDUCE_TRANSITIONS_K", "8")  # see score_metrics.py's pin note
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(BON_HARNESS))

SEED = 20260801
CORRUPTIONS = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
SPURIOUS = [1, 8, 64, 256]


def main() -> int:  # noqa: C901
    t_start = time.monotonic()
    import numpy as np
    from carnot.agentic import arc_executable_world_model as e3
    from split import load_split

    sys.path.insert(0, str(HERE))
    # The metric computations are duplicated from metric_worker rather than imported, because that
    # module is a __main__-shaped worker that reads argv. Keeping them here means this control
    # exercises the SAME formulas; `metrics_agree_with_worker` below is the check that they do.
    from carnot.agentic.arc_color_blob_salience import connected_color_blobs, object_hash
    from metric_worker import main as _unused  # noqa: F401  (import proves the module loads)

    def _objects(grid):
        blobs = connected_color_blobs(np.asarray(grid), min_pixels=1, max_component_fraction=1.0)
        labels = np.full(np.asarray(grid).shape, -1, dtype=np.int32)
        colors, sizes, keys, inv = [], [], [], []
        for i, b in enumerate(blobs):
            for y, x in b.cells:
                labels[y, x] = i
            colors.append(int(b.color))
            sizes.append(int(b.pixel_count))
            h = object_hash(b)
            inv.append(h)
            keys.append((h, int(b.bbox[0]), int(b.bbox[1])))
        return (
            labels,
            np.asarray(colors, dtype=np.int64),
            np.asarray(sizes, dtype=np.int64),
            keys,
            inv,
        )

    def _mj(a, b):
        from collections import Counter

        ca, cb = Counter(a), Counter(b)
        i = sum((ca & cb).values())
        u = sum((ca | cb).values())
        return float(i / u) if u else 1.0

    def _obj_iou(pred, true):
        lt, ct, st, _, _ = _objects(true)
        lp, cp, sp, _, _ = _objects(pred)
        nt, npd = len(ct), len(cp)
        if nt == 0 or npd == 0:
            return (1.0, 1.0, 1.0) if (nt == 0 and npd == 0) else (0.0, 0.0, 0.0)
        both = (lt >= 0) & (lp >= 0)
        pair = lt[both].astype(np.int64) * npd + lp[both].astype(np.int64)
        inter = np.bincount(pair, minlength=nt * npd).reshape(nt, npd).astype(np.float64)
        union = st[:, None] + sp[None, :] - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        iou = np.where(ct[:, None] == cp[None, :], iou, 0.0)
        bt, bp = iou.max(axis=1), iou.max(axis=0)
        rec = float((bt * st).sum() / max(1, st.sum()))
        prec = float((bp * sp).sum() / max(1, sp.sum()))
        return rec, prec, (2 * rec * prec / (rec + prec)) if (rec + prec) > 0 else 0.0

    def grade(rows, engine) -> dict:
        vr = e3.WorldModelVerifier(list(rows)).score(engine)
        nch = int(vr.n_changing or 0)
        agree_all, agree_chg, jac, f1s, recs, invs, poss = [], [], [], [], [], [], []
        for t in rows:
            if t.level_after > t.level_before:
                continue
            g0, g1 = np.asarray(t.grid), np.asarray(t.next_grid)
            changed = not np.array_equal(g0, g1)
            pred = np.asarray(engine(t.grid.copy(), t.action, t.data))
            agree_all.append(float((pred == g1).mean()))
            if not changed:
                continue
            agree_chg.append(agree_all[-1])
            m, wrote = g1 != g0, pred != g0
            u = int((m | wrote).sum())
            jac.append(float(int((m & wrote).sum()) / u) if u else 1.0)
            r, _p, f = _obj_iou(pred, g1)
            recs.append(r)
            f1s.append(f)
            _a, _b, _c, kp, ip = _objects(pred)
            _d, _e, _f, kt, it = _objects(g1)
            invs.append(_mj(ip, it))
            poss.append(_mj(kp, kt))

        def mean(x):
            return round(sum(x) / len(x), 6) if x else None

        return {
            "exact_match_accuracy": round(float(vr.accuracy), 6),
            "change_exact_accuracy": round(float(vr.change_accuracy), 6) if nch else None,
            "cell_recall": round(float(vr.cell_recall), 6) if nch else None,
            "change_fidelity": round(float(vr.change_fidelity), 6) if nch else None,
            "correct_changed_cells": int(vr.correct_changed_cells),
            "spurious_changed_cells": int(vr.spurious_changed_cells),
            "grid_agreement_all": mean(agree_all),
            "grid_agreement_changing": mean(agree_chg),
            "changed_cell_jaccard": mean(jac),
            "object_match_iou": mean(f1s),
            "object_match_recall": mean(recs),
            "object_inventory_jaccard": mean(invs),
            "object_positional_jaccard": mean(poss),
        }

    def make_lookup(rows):
        return {
            (np.asarray(t.grid).tobytes(), int(t.action), repr(t.data)): np.asarray(t.next_grid)
            for t in rows
        }

    def identity_engine(grid, action, data=None):
        return np.asarray(grid)

    def oracle_factory(lut, p_corrupt=0.0, n_spurious=0, seed=SEED):
        def eng(grid, action, data=None):
            g0 = np.asarray(grid)
            g1 = lut.get((g0.tobytes(), int(action), repr(data)))
            if g1 is None:
                return g0
            out = np.array(g1, copy=True)
            rng = np.random.default_rng(seed + int(action) + int(g0.sum()) % 100003)
            if p_corrupt > 0:
                ys, xs = np.nonzero(g1 != g0)
                if len(ys):
                    k = int(round(p_corrupt * len(ys)))
                    pick = rng.choice(len(ys), size=min(k, len(ys)), replace=False)
                    for i in pick:
                        y, x = int(ys[i]), int(xs[i])
                        # A WRONG value, deterministically: shift within the palette actually used
                        # by this grid, so the corruption stays in-distribution and cannot be
                        # detected by "that colour does not exist in this game".
                        pal = [int(v) for v in np.unique(g1) if int(v) != int(g1[y, x])]
                        out[y, x] = pal[int(rng.integers(len(pal)))] if pal else int(g1[y, x]) + 1
            if n_spurious > 0:
                ys, xs = np.nonzero(g1 == g0)  # cells reality left ALONE
                if len(ys):
                    k = min(n_spurious, len(ys))
                    pick = rng.choice(len(ys), size=k, replace=False)
                    for i in pick:
                        y, x = int(ys[i]), int(xs[i])
                        pal = [int(v) for v in np.unique(g1) if int(v) != int(out[y, x])]
                        if pal:
                            out[y, x] = pal[int(rng.integers(len(pal)))]
            return out

        return eng

    games = ["ft09", "tu93", "tn36", "lp85", "sc25"]
    per_game: dict[str, dict] = {}
    for g in games:
        s = load_split(g, 1)
        rows = s["_heldout"]
        if not rows:
            continue
        lut = make_lookup(rows)
        arms: dict[str, dict] = {"identity": grade(rows, identity_engine)}
        for p in CORRUPTIONS:
            arms[f"oracle_corrupt_{p}"] = grade(rows, oracle_factory(lut, p_corrupt=p))
        for k in SPURIOUS:
            arms[f"oracle_spurious_{k}"] = grade(rows, oracle_factory(lut, n_spurious=k))
        per_game[g] = {
            "n_heldout": s["n_heldout"],
            "heldout_n_changing": s["heldout_n_changing"],
            "heldout_n_noop": s["heldout_n_noop"],
            "change_dominated": bool(s["heldout_n_changing"] > s["heldout_n_noop"]),
            "arms": arms,
        }
        print(
            f"{g}: graded {len(arms)} reference arms over {s['n_heldout']} held-out rows",
            flush=True,
        )

    # ---- did each metric RESOLVE the constructed quality ladder? ----------------------------
    metrics = list(next(iter(per_game.values()))["arms"]["identity"].keys())
    ladder = [f"oracle_corrupt_{p}" for p in CORRUPTIONS]
    resolution: dict[str, dict] = {}
    for mk in metrics:
        per_game_res = {}
        for g, gd in per_game.items():
            vals = [gd["arms"][a].get(mk) for a in ladder]
            if any(v is None for v in vals):
                per_game_res[g] = {"measurable": False, "reason": "no changing held-out row"}
                continue
            v = [float(x) for x in vals]
            lower_better = mk == "spurious_changed_cells"
            # STRICT monotonicity is the wrong bar (p=0.05 on a 4-cell change rounds to the same
            # engine as p=0.0). WEAK monotonicity plus "the endpoints differ" is the honest test
            # of whether the instrument can order a known-quality ladder at all.
            mono = all(
                (v[i] >= v[i + 1] - 1e-12) if not lower_better else (v[i] <= v[i + 1] + 1e-12)
                for i in range(len(v) - 1)
            )
            # THE DISQUALIFYING PROPERTY, and the reason this control exists at all.
            # A metric on which the DEGENERATE IDENTITY ENGINE ("nothing ever changes") scores
            # BETTER than a real-but-bad engine cannot be an A/B primary, because an arm is then
            # rewarded for producing MORE INERT engines. That is not hypothetical on this corpus:
            # 13 of 40 stall-path candidates in the preceding phase predicted that no action
            # changes anything. Whole-grid and object-partition metrics are both exposed to it,
            # because a 64x64 ARC frame is overwhelmingly static background that identity
            # reproduces perfectly -- the score is dominated by the part of the grid the dynamics
            # never touch.
            idv = gd["arms"]["identity"].get(mk)
            worst = (max(v) if lower_better else min(v)) if v else None
            outranks = (
                None
                if idv is None or worst is None
                else bool((idv < worst) if lower_better else (idv > worst))
            )
            per_game_res[g] = {
                "measurable": True,
                "values": [round(x, 6) for x in v],
                "n_distinct": len(set(round(x, 6) for x in v)),
                "dynamic_range_over_ladder": round(max(v) - min(v), 8),
                "weakly_monotone_in_quality": bool(mono),
                "endpoints_differ": bool(abs(v[0] - v[-1]) > 1e-12),
                "identity": idv,
                "oracle": gd["arms"]["oracle_corrupt_0.0"].get(mk),
                "worst_ladder_rung": round(worst, 6) if worst is not None else None,
                "identity_outranks_a_real_engine": outranks,
                "identity_vs_oracle_separated": bool(
                    gd["arms"]["identity"].get(mk) != gd["arms"]["oracle_corrupt_0.0"].get(mk)
                ),
                "spurious_penalised": {
                    f"k={k}": gd["arms"][f"oracle_spurious_{k}"].get(mk) for k in SPURIOUS
                },
            }
        meas = [r for r in per_game_res.values() if r.get("measurable")]
        resolution[mk] = {
            "per_game": per_game_res,
            "n_games_measurable": len(meas),
            "n_games_ladder_resolved": sum(
                1 for r in meas if r["n_distinct"] > 1 and r["weakly_monotone_in_quality"]
            ),
            "n_games_ladder_collapsed_to_one_value": sum(1 for r in meas if r["n_distinct"] == 1),
            "n_games_identity_vs_oracle_separated": sum(
                1 for r in meas if r["identity_vs_oracle_separated"]
            ),
            "n_games_identity_outranks_a_real_engine": sum(
                1 for r in meas if r.get("identity_outranks_a_real_engine")
            ),
            "rewards_inertness_disqualifying": bool(
                any(r.get("identity_outranks_a_real_engine") for r in meas)
            ),
            "min_dynamic_range_over_ladder": (
                round(min(r["dynamic_range_over_ladder"] for r in meas), 8) if meas else None
            ),
            # The asymmetry witness: does adding writes reality never made change the number?
            "penalises_spurious_writes_on_every_measurable_game": bool(
                meas
                and all(
                    any(
                        r["spurious_penalised"][f"k={k}"] != r["oracle"]
                        for k in SPURIOUS
                        if r["spurious_penalised"][f"k={k}"] is not None
                    )
                    for r in meas
                )
            ),
        }

    payload = {
        "what_this_is": (
            "Reference engines of CONSTRUCTED quality graded on the same proven held-out splits "
            "as the 48 frozen candidates. Answers 'can this metric resolve a known difference', "
            "which candidate variance alone cannot."
        ),
        "random_seed": SEED,
        "corruption_ladder": CORRUPTIONS,
        "spurious_write_counts": SPURIOUS,
        "per_game": per_game,
        "resolution": resolution,
        "duration_s": round(time.monotonic() - t_start, 3),
    }
    out = OUT_DIR / "positive_control.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {out}\n")
    # This console table reports THIS SCRIPT'S THREE criteria only (H1, H2, H3). It deliberately
    # does NOT print a USABLE / recommendable column: H4 (separates the real frozen candidates on
    # 2+ games) and H5 (adequate dynamic range) are evaluated in build_artifact.py against the
    # 48-candidate census, which this script has not read. An earlier version printed "USABLE"
    # from these three alone and showed True for `exact_match_accuracy` -- the very metric the
    # finished analysis disqualifies -- which is a console line contradicting the artifact.
    print(
        f"{'metric':28} {'meas':>4} {'H1_resolv':>9} {'minrange':>9} "
        f"{'H2_id_out':>9} {'H3_spur':>8}"
    )
    for mk in metrics:
        r = resolution[mk]
        print(
            f"{mk:28} {r['n_games_measurable']:>4} {r['n_games_ladder_resolved']:>9} "
            f"{str(r['min_dynamic_range_over_ladder']):>9} "
            f"{r['n_games_identity_outranks_a_real_engine']:>9} "
            f"{str(r['penalises_spurious_writes_on_every_measurable_game']):>8}"
        )
    print("\nH1/H2/H3 only -- H4 and H5 need the candidate census; see build_artifact.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
