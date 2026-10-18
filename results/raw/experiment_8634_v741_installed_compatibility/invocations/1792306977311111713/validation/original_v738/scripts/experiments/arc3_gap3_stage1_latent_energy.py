"""GAP-3 Stage 1: does TRM's FULL penultimate latent (z_H[:,0], hidden=512) -- the un-collapsed vector
the q_head projects into the scalar q_halt -- beat frequency vote as a model-native selection energy?

Stage 0 showed the SCALAR q_halt does NOT beat vote (results/arc3_gap3_stage0_qhalt_energy.json,
adversarially confirmed) but has soft within-task AUROC 0.86 -- the signal EXISTS in TRM's confidence,
the 1-D projection just throws it away. Stage 1 tests the FULL latent. Design:
docs/research-notes/gap3-learned-arc-energy-design.md Stage 1 (model-native basis 1a + learned probe 1b).

Substrate: a FRESH capped TRM eval dump that also saves z_h_pool (the penultimate latent), produced by
  trm_arc_eval_harness.py --save_latent  (writes eval_out/arc_v1_latent/step_0_all_preds.*).

Energy (model-native, oracle-free at inference; LABEL used ONLY out-of-fold for training the probe):
  * per candidate: z_mean = mean of z_H[:,0] over the augmentations that produced it (the candidate's
    representative latent), plus votes + q_mean (Stage-0 features, for the baselines).
  * LEAVE-ONE-TASK-OUT: for each held-out task, fit a compact PCA BASIS (top-K components, the
    model-native orthogonal basis of arXiv:2604.17614) on the OTHER tasks' candidate latents, project,
    then fit a class-balanced L2 logistic probe basis-coords -> {gold=keep, non-gold=prune}. Predict the
    held-out task's candidates OUT-OF-FOLD. Rank by prob ascending (lower = more gold-like). PCA fit on
    train folds only -> no test latent leaks into the basis; labels never seen for the held-out task.

Rankers: TRM_VOTE (baseline), Q_MEAN (Stage-0 scalar), LATENT_PROBE (Stage 1), and HYBRID(vote primary,
latent-probe tie-break). A0 vote-mimicry control: residualize the probe over vote (does it beat vote
with vote regressed out?) + within-task AUROC of the probe + bootstrap CI of (probe - vote).

Gates (design Section 3): selection probe pass@2 > vote; within-task AUROC > 0.70; coverage 100% (latent
always defined); headroom-capture >= 30% of (oracle - vote). NO ORACLE at inference: gold labels train
the probe OUT-OF-FOLD only; the held-out task's gold is never seen when ranking it.

CAVEAT: ~31-task subset; a 512-d latent with ~31 LOTO folds overfits easily -> PCA-K small + L2 + balanced
classes are the guards, and the OOF protocol is the honesty anchor. Report as-is per FALSE_NEGATIVE_RISK.

  ~/trm_venv/bin/python scripts/experiments/arc3_gap3_stage1_latent_energy.py [--pca_k 24]
"""

from __future__ import annotations

import argparse
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
LATENT_GLOB = f"{TRM}/eval_out/arc_v1_latent/step_0_all_preds.*"

import torch  # noqa: E402
from evaluators.arc import ARC, _crop  # noqa: E402  (reuse the published de-aug crop)
from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np  # noqa: E402


def _auroc(pos, neg):
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _standardize(X):
    mean = X.mean(0)
    std = X.std(0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def _fit_logreg_weighted(X, y, w_pos, w_neg, l2=1.0, lr=0.2, iters=300):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    yv = np.asarray(y, dtype=np.float64)
    cw = np.where(yv == 1, w_pos, w_neg)
    wsum = cw.sum()
    for _ in range(iters):
        z = b + X @ w
        z = np.clip(z, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        err = cw * (p - yv)
        gb = err.sum()
        gw = X.T @ err
        b -= lr * (gb / wsum)
        w -= lr * (gw / wsum + l2 * w / n)
    return w, b


def _predict_proba(w, b, X):
    z = np.clip(b + X @ w, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def _build_candidate_table_with_latent():
    """Replicate the ARC evaluator's update_batch de-aug per row, but carry z_h_pool, so each candidate
    gets votes + q_mean + z_mean. Returns list of {task, cands:[{votes,q_mean,z_mean,correct}]}."""
    ev = ARC(data_path=DATA, eval_metadata=SimpleNamespace(blank_identifier_id=0))
    shards = sorted(glob.glob(LATENT_GLOB))
    if not shards:
        return None, ev
    # accumulate per (orig_task, input_hash): {pred_hash: {votes, qs[], zs[]}}, and pred_hash->grid
    store = {}
    for sh in shards:
        d = torch.load(sh, map_location="cpu")
        pids = d["puzzle_identifiers"].numpy()
        inputs = d["inputs"].numpy()
        preds = d["preds"].numpy()
        qh = d["q_halt_logits"].to(torch.float64).sigmoid().numpy()
        z = d["z_h_pool"].to(torch.float32).numpy()  # (rows, hidden)
        keep = pids != ev.blank_identifier_id
        for i in np.nonzero(keep)[0]:
            name = ev.identifier_map[int(pids[i])]
            orig_name, inv_fn = inverse_aug(name)
            ih = grid_hash(inv_fn(_crop(inputs[i])))
            pg = inv_fn(_crop(preds[i]))
            ph = grid_hash(pg)
            slot = (
                store.setdefault(orig_name, {})
                .setdefault(ih, {})
                .setdefault(ph, {"votes": 0, "qs": [], "zs": []})
            )
            slot["votes"] += 1
            slot["qs"].append(float(qh[i]))
            slot["zs"].append(z[i])
    # join to test_puzzles to get gold hash per (task, test input)
    tasks = []
    for name, puzzle in ev.test_puzzles.items():
        if name not in store:
            continue
        for pair in puzzle["test"]:
            ih = grid_hash(arc_grid_to_np(pair["input"]))
            lh = grid_hash(arc_grid_to_np(pair["output"]))
            bucket = store[name].get(ih)
            if not bucket:
                continue
            cands = []
            for ph, s in bucket.items():
                cands.append(
                    {
                        "votes": s["votes"],
                        "q_mean": float(np.mean(s["qs"])),
                        "z_mean": np.mean(np.stack(s["zs"]), axis=0),
                        "correct": bool(ph == lh),
                    }
                )
            if cands:
                tasks.append({"task": name, "cands": cands})
    return tasks, ev


def _pass(tasks, key, ks=(1, 2)):
    hits = {k: 0 for k in ks}
    for t in tasks:
        ranked = sorted(t["cands"], key=key)
        rc = [c["correct"] for c in ranked]
        for k in ks:
            hits[k] += int(any(rc[:k]))
    n = len(tasks)
    return {f"pass@{k}": round(hits[k] / n, 4) for k in ks}


def _loto_latent_probe(tasks, pca_k):
    """Leave-one-task-out: PCA basis (fit on train tasks) -> balanced logistic probe -> OOF prob for the
    held-out task's candidates. Writes c['probe'] in place. No label or latent of the held task is used
    to fit its own basis/probe."""
    for held in range(len(tasks)):
        train_cands = [c for j, t in enumerate(tasks) if j != held for c in t["cands"]]
        test_cands = tasks[held]["cands"]
        Xtr = np.stack([c["z_mean"] for c in train_cands])
        mu = Xtr.mean(0)
        Xtrc = Xtr - mu
        # PCA basis (top-K right singular vectors) fit on TRAIN candidate latents only
        k = min(pca_k, Xtrc.shape[1], max(1, Xtrc.shape[0] - 1))
        _, _, Vt = np.linalg.svd(Xtrc, full_matrices=False)
        basis = Vt[:k]  # (k, hidden)
        Ztr = Xtrc @ basis.T  # (Ntr, k)
        Zs, smean, sstd = _standardize(Ztr)
        ytr = np.array([0 if c["correct"] else 1 for c in train_cands])  # 1 = prune
        npos = max(1, int(ytr.sum()))
        nneg = max(1, len(ytr) - npos)
        w_pos = len(ytr) / (2.0 * npos)
        w_neg = len(ytr) / (2.0 * nneg)
        w, b = _fit_logreg_weighted(Zs, ytr, w_pos, w_neg)
        for c in test_cands:
            zc = (((c["z_mean"] - mu) @ basis.T) - smean) / sstd
            c["probe"] = float(_predict_proba(w, b, zc[None, :])[0])


def run(pca_k=24, write=True):
    started = time.time()
    tasks, ev = _build_candidate_table_with_latent()
    if not tasks:
        art = {
            "experiment": "arc3_gap3_stage1_latent_energy",
            "honest_verdict": "blocked_no_latent_dump",
            "inference_substrate": "offline_trm_candidate_rerank_no_oracle",
            "note": f"no latent dump at {LATENT_GLOB}; run trm_arc_eval_harness.py --save_latent first",
        }
        if write:
            Path(f"{CARNOT}/results/arc3_gap3_stage1_latent_energy.json").write_text(
                json.dumps(art, indent=2) + "\n"
            )
        print("-> blocked_no_latent_dump")
        return art

    n = len(tasks)
    n_oracle = sum(1 for t in tasks if any(c["correct"] for c in t["cands"]))
    hidden = int(tasks[0]["cands"][0]["z_mean"].shape[0])

    _loto_latent_probe(tasks, pca_k)

    # ---- A0 control: residualize the probe over vote (regress globally, rank by residual) ----
    allc = [c for t in tasks for c in t["cands"]]
    vs = np.array([c["votes"] for c in allc], dtype=float)
    pr = np.array([c["probe"] for c in allc], dtype=float)
    b = np.cov(vs, pr)[0, 1] / (vs.var() or 1.0)
    a = pr.mean() - b * vs.mean()
    for c in allc:
        c["_probe_resid"] = c["probe"] - (a + b * c["votes"])

    rankers = {
        "TRM_VOTE": lambda c: (-c["votes"],),
        "Q_MEAN": lambda c: (-c["q_mean"], -c["votes"]),
        "LATENT_PROBE": lambda c: (c["probe"], -c["votes"]),  # lower prob(prune) = better
        "LATENT_PROBE_residual_over_vote": lambda c: (c["_probe_resid"], -c["votes"]),
        "HYBRID_vote_then_probe": lambda c: (-c["votes"], c["probe"]),
    }
    res = {name: _pass(tasks, key) for name, key in rankers.items()}
    oracle2 = round(n_oracle / n, 4)

    # within-task AUROC of the probe (feed -probe so >0.5 = gold has LOWER prune-prob = better)
    per_task_auroc = []
    for t in tasks:
        g = [-c["probe"] for c in t["cands"] if c["correct"]]
        ng = [-c["probe"] for c in t["cands"] if not c["correct"]]
        au = _auroc(g, ng)
        if au is not None:
            per_task_auroc.append(au)
    probe_auroc_within = (
        round(sum(per_task_auroc) / len(per_task_auroc), 4) if per_task_auroc else None
    )

    # bootstrap-over-tasks 95% CI on pass@2(LATENT_PROBE) - pass@2(vote), deterministic LCG
    def _lcg(seed):
        x = seed
        while True:
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            yield x

    gen = _lcg(12345)
    B = 1000

    def _p2(sample_tasks, key):
        return sum(
            int(any(c["correct"] for c in sorted(t["cands"], key=key)[:2])) for t in sample_tasks
        ) / len(sample_tasks)

    kp, kv = rankers["LATENT_PROBE"], rankers["TRM_VOTE"]
    deltas = []
    for _ in range(B):
        idx = [next(gen) % n for _ in range(n)]
        samp = [tasks[i] for i in idx]
        deltas.append(_p2(samp, kp) - _p2(samp, kv))
    deltas.sort()
    ci = [round(deltas[int(0.025 * B)], 4), round(deltas[int(0.975 * B)], 4)]
    delta_point = round(res["LATENT_PROBE"]["pass@2"] - res["TRM_VOTE"]["pass@2"], 4)

    vote2 = res["TRM_VOTE"]["pass@2"]
    sel_pass = res["LATENT_PROBE"]["pass@2"] > vote2
    resid_beats_vote = res["LATENT_PROBE_residual_over_vote"]["pass@2"] > vote2
    headroom = (res["LATENT_PROBE"]["pass@2"] - vote2) / max(1e-9, oracle2 - vote2)
    gates = {
        "selection_beats_vote": bool(sel_pass),
        "discrimination_within_task_auroc_gt_0p70": bool((probe_auroc_within or 0) > 0.70),
        "coverage_ge_0p80": True,
        "headroom_capture_ge_0p30": bool(headroom >= 0.30),
        "headroom_capture_fraction": round(headroom, 4),
    }
    model_native_real = bool(sel_pass and resid_beats_vote)

    verdict = (
        "complete: gap3_stage1_latent_"
        + (
            "full_latent_BEATS_vote_model_native_energy_works"
            if model_native_real
            else (
                "latent_probe_beats_vote_but_vote_redundant"
                if sel_pass
                else "full_latent_does_not_beat_vote"
            )
        )
        + f"_n{n}_vote_{vote2}_probe_{res['LATENT_PROBE']['pass@2']}"
        + f"_withinauroc_{probe_auroc_within}_residbeatsvote_{resid_beats_vote}_pcak{pca_k}"
    )

    art = {
        "experiment": "arc3_gap3_stage1_latent_energy",
        "title": "GAP-3 Stage 1: TRM penultimate-latent model-native selection energy vs frequency vote",
        "honest_verdict": verdict,
        "inference_substrate": "offline_trm_candidate_rerank_no_oracle",
        "n_tasks": n,
        "n_oracle_hit": n_oracle,
        "oracle_pass2_ceiling": oracle2,
        "latent_hidden_dim": hidden,
        "pca_k": pca_k,
        "rankers": res,
        "model_native_signal_real": model_native_real,
        "a0_vote_mimicry_control": {
            "residual_ranker_pass2": res["LATENT_PROBE_residual_over_vote"]["pass@2"],
            "residual_beats_vote": bool(resid_beats_vote),
            "probe_within_task_auroc": probe_auroc_within,
            "note": (
                "LATENT_PROBE is an OUT-OF-FOLD leave-one-task-out PCA-basis + balanced logistic probe; "
                "the held-out task's gold labels and latents never train its own probe/basis. residual "
                "regresses the probe on vote-count globally and ranks by the residual -- if it does not "
                "beat vote, the latent's lift is vote-collinear, not orthogonal model-native signal."
            ),
        },
        "gates": gates,
        "bootstrap_pass2_delta_vs_vote": {"point": delta_point, "ci95_over_tasks": ci, "B": B},
        "vs_stage0": (
            "Stage 0 (scalar q_halt) did NOT beat vote (Q_MEAN pass@2 0.290; residual 0.097). Stage 1 "
            "tests whether the FULL un-collapsed latent (hidden-dim) recovers the signal the scalar lost."
        ),
        "no_oracle_audit": (
            "Gold output read ONLY to LABEL correctness; the probe is trained OUT-OF-FOLD (LOTO) so a "
            "held-out task's gold never trains its own ranker; PCA basis fit on train folds only."
        ),
        "honest_note": (
            "GAP-3 Stage 1. A POSITIVE that survives the residual control = the model-native energy works "
            "-> register it + re-confirm at 400 scale. A NEGATIVE = even the full latent can't beat vote "
            "on this corpus (deeper: the headroom may need a generator-INDEPENDENT trained ARC energy, "
            "Stage 2). Reported as-is per FALSE_NEGATIVE_RISK; positive control = oracle > vote (real "
            "headroom). CAVEAT n~31 + 512-d latent over ~31 LOTO folds overfits easily; PCA-K + L2 + "
            "balanced classes + OOF are the guards. Re-confirm at 400 before any irreversible claim."
        ),
        "no_gpu_used_for_this_step": True,
        "random_seed": 12345,
        "duration_s": round(time.time() - started, 1),
    }
    if write:
        Path(f"{CARNOT}/results/arc3_gap3_stage1_latent_energy.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n"
        )
    print(f"-> {verdict}")
    print(
        f"   n_tasks={n} oracle_hit={n_oracle} oracle_pass2={oracle2} hidden={hidden} pca_k={pca_k}"
    )
    for r in [
        "TRM_VOTE",
        "Q_MEAN",
        "LATENT_PROBE",
        "LATENT_PROBE_residual_over_vote",
        "HYBRID_vote_then_probe",
    ]:
        print(f"   {r:32s} pass@1={res[r]['pass@1']} pass@2={res[r]['pass@2']}")
    print(
        f"   A0: resid_beats_vote={resid_beats_vote} within_auroc={probe_auroc_within} -> model_native_real={model_native_real}"
    )
    print(f"   gates={gates}")
    print(f"   bootstrap probe-vote pass@2: point={delta_point} CI95={ci}")
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca_k", type=int, default=24)
    a = ap.parse_args()
    run(pca_k=a.pca_k)
