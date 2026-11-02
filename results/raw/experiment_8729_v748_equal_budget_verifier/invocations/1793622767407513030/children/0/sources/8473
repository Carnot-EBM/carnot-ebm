"""HAND-FEATURE CEILING: can a LEARNED combiner over the 8 cheap families beat TRM's vote on TRM's
real candidate pool? (The definitive "are hand-invariants salvageable?" test, per operator directive.)

Context. arc3_verifier_antirank_diagnosis.json showed the union_max (max) aggregation anti-ranks
(pass@2 0.19) because it takes each candidate's WORST family and two families (v1 AUROC 0.42,
delta_pattern 0.43) are anti-discriminative on TRM's real candidates — while color_mapping (0.71),
object_count (0.67), content_overlap (0.67), tiling (0.91) carry real but low-coverage signal. min()
recovers to ~vote (0.42) but no fixed aggregation BEATS vote (0.45). The open question: does a LEARNED
linear combiner — which can down-weight v1/delta_pattern and up-weight the discriminative families —
beat vote? If not, the cheap hand-features are exhausted and only a learned/model-native ARC energy
(GAP-3) can reach the proven ~13pp headroom (oracle 0.61 > vote 0.45).

Method (NO oracle in the ranker; the LABEL trains the combiner out-of-fold only):
  * Per TRM task, each unique candidate -> feature row = V._combined_scores feats [v1, symmetry,
    color_mapping, object_count, content_overlap, delta_pattern, tiling_scaling, palette_histogram];
    label 1 = NON-gold (prune), 0 = gold (keep); group = task (so a task's candidates never split CV
    folds -> no within-task leakage).
  * Two combiners, both reporting OUT-OF-FOLD probabilities (no in-sample optimism):
      - logreg_5fold_groupcv  : reuse V._logistic_cv_oof (the v2 ceiling estimator), unweighted.
      - logreg_loto_balanced  : leave-one-TASK-out CV + class-balanced weights (gold is 1/N rare per
                                task; balancing stops the combiner from learning "everything is a
                                distractor"). The honest small-sample ceiling for 31 tasks.
  * Rank candidates within task by oof prob ASCENDING (lower prob = more gold-like = keep); pass@1/2.
  * Report learned weights (averaged across folds) to show WHICH families the combiner trusts.
  * Compare to TRM_VOTE and min_defined (the diagnosis baselines).

Honest caveat: 31 tasks / 19 with gold is a TINY training set; a learned combiner can over/under-fit.
The out-of-fold protocol + class balancing are the honest guards. Report whatever it shows, including
"no learned combiner beats vote" (that is the load-bearing GAP-3 trigger, per FALSE_NEGATIVE_RISK).

  ~/trm_venv/bin/python scripts/experiments/arc3_verifier_learned_combiner_ceiling.py
"""

from __future__ import annotations

import glob
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from carnot.paths import repo_root

# Resolved via the central resolver rather than hardcoded: a hardcoded
# absolute path makes a fresh clone write into the original author's
# checkout. See python/carnot/paths.py.
CARNOT = str(repo_root())
TRM = "/home/ianblenke/trm_src"
sys.path.insert(0, f"{CARNOT}/scripts/experiments")
sys.path.insert(0, TRM)

DATA = f"{TRM}/data/arc1concept-aug-1000"
PREDS_GLOB = f"{TRM}/eval_out/arc_v1/step_0_all_preds.*"

import torch  # noqa: E402  (trm_venv)
from evaluators.arc import ARC  # noqa: E402
from dataset.build_arc_dataset import arc_grid_to_np, grid_hash  # noqa: E402
import arc_grid_verifier_invariants_v2_combined as V  # noqa: E402,N812

FEAT_NAMES = ["v1", *V.FAMILY_ORDER]


def _as_list(g):
    return g.tolist() if isinstance(g, np.ndarray) else g


def _fit_logreg_weighted(X, y, w_pos, w_neg, l2=1.0, lr=0.2, iters=400):
    """Class-weighted L2 logistic (gradient descent). w_pos weights label-1 (prune) rows, w_neg the
    label-0 (gold/keep) rows, so the rare gold class is not drowned out by the many non-gold rows."""
    n, d = len(X), len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(iters):
        gw = [0.0] * d
        gb = 0.0
        wsum = 0.0
        for i in range(n):
            z = b + sum(w[j] * X[i][j] for j in range(d))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            cw = w_pos if y[i] == 1 else w_neg
            err = cw * (p - y[i])
            gb += err
            wsum += cw
            for j in range(d):
                gw[j] += err * X[i][j]
        b -= lr * (gb / wsum)
        for j in range(d):
            w[j] -= lr * (gw[j] / wsum + l2 * w[j] / n)
    return w, b


def _loto_balanced_oof(rows):
    """rows: (feat_vec, label, group_idx). Leave-one-TASK-out CV, class-balanced. Returns (oof_prob
    per row, avg standardized weights dict)."""
    groups = sorted({r[2] for r in rows})
    oof = [None] * len(rows)
    wsum = [0.0] * len(FEAT_NAMES)
    nfolds = 0
    for held in groups:
        tr = [(r[0], r[1]) for r in rows if r[2] != held]
        teidx = [i for i, r in enumerate(rows) if r[2] == held]
        if not tr or not teidx:
            continue
        Xtr = [a for a, _ in tr]
        ytr = [bb for _, bb in tr]
        npos = sum(ytr) or 1
        nneg = (len(ytr) - sum(ytr)) or 1
        # balance: weight each class inversely to its frequency
        w_pos = len(ytr) / (2.0 * npos)
        w_neg = len(ytr) / (2.0 * nneg)
        Z, mean, std = V._standardize(Xtr)
        w, b = _fit_logreg_weighted(Z, ytr, w_pos, w_neg)
        for j in range(len(FEAT_NAMES)):
            wsum[j] += w[j]
        nfolds += 1
        for i in teidx:
            xz = [(rows[i][0][j] - mean[j]) / std[j] for j in range(len(mean))]
            oof[i] = V._predict_proba(w, b, xz)
    avg_w = (
        {FEAT_NAMES[j]: round(wsum[j] / nfolds, 4) for j in range(len(FEAT_NAMES))}
        if nfolds
        else {}
    )
    return oof, avg_w


def run(write=True):
    started = time.time()
    ev = ARC(data_path=DATA, eval_metadata=SimpleNamespace(blank_identifier_id=0))
    for sh in sorted(glob.glob(PREDS_GLOB)):
        d = torch.load(sh, map_location="cpu")
        ev.update_batch(
            {"inputs": d["inputs"], "puzzle_identifiers": d["puzzle_identifiers"]},
            {"preds": d["preds"], "q_halt_logits": d["q_halt_logits"]},
        )

    # build per-task candidate tables + a flat row table for the learned combiners
    tasks = []  # list of {cands:[...], gold_ph}
    log_rows = []  # (feat_vec, label, group_idx, kind, is_gold)  -- V._logistic_cv_oof format
    loto_rows = []  # (feat_vec, label, group_idx)
    row_index = {}  # (group_idx, ph) -> flat row index (shared order for both)
    gi = 0
    for name, puzzle in ev.test_puzzles.items():
        if name not in ev._local_preds:
            continue
        inv = V._build_invariants(puzzle["train"])
        for pair in puzzle["test"]:
            ih = grid_hash(arc_grid_to_np(pair["input"]))
            lh = grid_hash(arc_grid_to_np(pair["output"]))
            preds = ev._local_preds[name].get(ih)
            if not preds:
                continue
            cand = {}
            for ph, q in preds:
                c = cand.setdefault(ph, [0, 0.0])
                c[0] += 1
                c[1] += q
            til = _as_list(arc_grid_to_np(pair["input"]))
            app = V._applicability(inv, puzzle["train"], til)
            cands = []
            for ph, (cnt, sq) in cand.items():
                try:
                    _um, _md, feats = V._combined_scores(
                        _as_list(ev._local_hmap[ph]), inv, puzzle["train"], til, app
                    )
                except Exception:
                    continue
                fv = [float(x) for x in feats]
                is_gold = ph == lh
                ridx = len(log_rows)
                row_index[(gi, ph)] = ridx
                log_rows.append(
                    (fv, 0 if is_gold else 1, gi, "GOLD" if is_gold else "PRED", is_gold)
                )
                loto_rows.append((fv, 0 if is_gold else 1, gi))
                applied = [nm for nm in FEAT_NAMES if (nm == "v1" or app.get(nm, False))]
                mindef = min(feats[FEAT_NAMES.index(nm)] for nm in applied)
                cands.append(
                    {
                        "ph": ph,
                        "votes": cnt,
                        "ridx": ridx,
                        "correct": is_gold,
                        "min_defined": float(mindef),
                    }
                )
            if cands:
                tasks.append({"cands": cands})
                gi += 1

    n_tasks = len(tasks)
    n_oracle = sum(1 for t in tasks if any(c["correct"] for c in t["cands"]))

    # ---- learned combiners (out-of-fold) ----
    oof_5fold = V._logistic_cv_oof(log_rows, k=5, seed=0)
    oof_loto, avg_w = _loto_balanced_oof(loto_rows)

    def _pass_at(ranker_key):
        hit1 = hit2 = 0
        for t in tasks:
            ranked = sorted(t["cands"], key=ranker_key)
            rc = [c["correct"] for c in ranked]
            hit1 += int(any(rc[:1]))
            hit2 += int(any(rc[:2]))
        return {"pass@1": round(hit1 / n_tasks, 4), "pass@2": round(hit2 / n_tasks, 4)}

    res = {
        "TRM_VOTE": _pass_at(lambda c: -c["votes"]),
        "min_defined": _pass_at(lambda c: (c["min_defined"], -c["votes"])),
        "logreg_5fold_groupcv": _pass_at(
            lambda c: (
                oof_5fold[c["ridx"]] if oof_5fold[c["ridx"]] is not None else 1.0,
                -c["votes"],
            )
        ),
        "logreg_loto_balanced": _pass_at(
            lambda c: (oof_loto[c["ridx"]] if oof_loto[c["ridx"]] is not None else 1.0, -c["votes"])
        ),
    }

    vote2 = res["TRM_VOTE"]["pass@2"]
    learned = {"logreg_5fold_groupcv", "logreg_loto_balanced"}
    beats = [r for r in learned if res[r]["pass@2"] > vote2]
    verdict = (
        "complete: learned_combiner_ceiling_"
        + (
            "BEATS_vote_" + max(learned, key=lambda r: res[r]["pass@2"])
            if beats
            else "no_learned_combiner_beats_vote_handfeatures_exhausted"
        )
        + f"_n{n_tasks}_vote_{vote2}"
        + f"_loto_{res['logreg_loto_balanced']['pass@2']}_5fold_{res['logreg_5fold_groupcv']['pass@2']}"
    )

    art = {
        "experiment": "arc3_verifier_learned_combiner_ceiling",
        "title": "Learned-combiner ceiling for the cheap hand-features on TRM's real candidate pool",
        "honest_verdict": verdict,
        "inference_substrate": "offline_trm_candidate_rerank_no_oracle",
        "n_tasks": n_tasks,
        "n_oracle_hit": n_oracle,
        "rankers": res,
        "loto_balanced_avg_standardized_weights": avg_w,
        "weights_note": (
            "Avg per-fold standardized logistic weights (label 1 = prune). POSITIVE weight => higher "
            "family score pushes 'prune' (so the combiner trusts the family to flag wrong candidates); "
            "weight near 0 => ignored; NEGATIVE => the combiner learned the family is anti-discriminative "
            "and inverts it. Expect v1/delta_pattern near 0 or negative, color_mapping/object_count/"
            "tiling positive (per the diagnosis AUROCs)."
        ),
        "beats_vote": beats,
        "honest_note": (
            "DEFINITIVE hand-feature ceiling (operator: diagnose before building). If beats_vote is "
            "empty, even an out-of-fold learned linear combiner over the 8 cheap families cannot beat "
            "TRM's frequency vote on its real candidate pool => the cheap hand-features are exhausted "
            "and GAP-3 (learned/model-native ARC energy from TRM's local activations) is the only path "
            "to the proven oracle headroom. Caveat: 31-task / 19-gold sample is tiny; out-of-fold + "
            "class-balanced LOTO are the honest guards but the number is noisy — re-confirm at scale "
            "before any irreversible strategy commitment."
        ),
        "no_gpu_used": True,
        "random_seed": 0,
        "duration_s": round(time.time() - started, 1),
    }
    if write:
        Path(f"{CARNOT}/results/arc3_verifier_learned_combiner_ceiling.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n"
        )

    print(f"-> {verdict}")
    print(f"   n_tasks={n_tasks} oracle_hit={n_oracle}")
    for r, v in res.items():
        flag = "  <-- BEATS vote" if r in beats else ""
        print(f"   {r:24s} pass@1={v['pass@1']} pass@2={v['pass@2']}{flag}")
    print("   LOTO balanced avg weights (label1=prune; >0 trust, <0 anti):")
    for nm in FEAT_NAMES:
        print(f"     {nm:24s} {avg_w.get(nm)}")
    return art


if __name__ == "__main__":
    run()
