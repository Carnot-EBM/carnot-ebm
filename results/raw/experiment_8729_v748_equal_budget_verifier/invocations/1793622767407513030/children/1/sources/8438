"""GAP-3 Stage 0: is TRM's OWN halting-confidence (q_halt) a model-native selection energy that beats
its frequency vote on its real candidate pool? (Zero-GPU; the dump already holds q_halt_logits.)

Design: docs/research-notes/gap3-learned-arc-energy-design.md Stage 0. The cheapest possible
model-native probe: TRM's published ARC pipeline selects by augmentation-FREQUENCY vote and IGNORES the
halting head for selection (it only uses q_halt to STOP refinement). We have q_halt per augmentation in
the offline dump. Question: does aggregating q_halt per de-augmented candidate re-rank better than vote
-- and is any lift REAL (model-native) or just collinear with vote-count (the A0 vote-mimicry trap)?

Energy: E(candidate) = -agg(q_halt over the augmentations that produced it). Lower E = more confident =
better. Aggregations:
  * q_mean  : mean q_halt          (vote-DEcorrelated -- the honest model-native signal)
  * q_max   : max  q_halt          (vote-DEcorrelated)
  * q_lse   : logsumexp q_halt     (vote-CONFOUNDED by construction: more augs -> higher lse; reported
                                    as the confounded reference, NOT a clean model-native signal)

Rankers compared on pass@1/2 (correct answer present in pool): TRM_VOTE (baseline), Q_MEAN, Q_MAX,
Q_LSE, HYBRID (vote primary + q_mean tie-break), plus the hand-feature baselines min_defined / union_max
for context.

A0 ADVERSARIAL CONTROL (vote-mimicry, the pivotal check): the lift of q_mean must be measured OVER vote,
not absolute. We compute (a) within-task Spearman(q_mean, vote), (b) a RESIDUAL ranker: globally regress
q_mean on vote (linear), rank by the residual q_mean - (a + b*vote); if the residual ranker carries NO
selection signal, q_mean's lift is just frequency in disguise. (c) discrimination AUROC(q_mean: gold vs
non-gold within task).

Gates (design Section 3): selection pass@2 > vote; discrimination AUROC > 0.70; coverage (q_halt defined
for 100% of candidates, trivially); headroom-capture >= 30% of (oracle - vote). Plus a bootstrap-over-
tasks 95% CI on the pass@2 delta vs vote (n=31 is tiny -> report the CI, do not over-claim).

NO ORACLE in any ranker: the gold output is read ONLY to LABEL which candidate is correct for scoring;
q_halt comes from TRM's forward pass, never from the gold. Emits a COMPACT per-candidate table
(results/arc3_gap3_stage0_candidate_table.json) so the finding can be independently re-derived WITHOUT
re-loading the 270MB torch dump (for the adversarial-verify workflow).

  ~/trm_venv/bin/python scripts/experiments/arc3_gap3_stage0_qhalt_energy.py
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


def _logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _auroc(pos, neg):
    """P(pos fed-value > neg fed-value), ties=0.5."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _spearman(x, y):
    """Spearman rho via Pearson on ranks (no scipy)."""
    if len(x) < 2:
        return None

    def _ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else None


def _build_candidate_table():
    """Replay the dump ONCE -> per-(task,test) list of candidates with votes + full q_halt list +
    hand-feature scores + correctness label. The correctness label uses gold ONLY for scoring."""
    ev = ARC(data_path=DATA, eval_metadata=SimpleNamespace(blank_identifier_id=0))
    for sh in sorted(glob.glob(PREDS_GLOB)):
        d = torch.load(sh, map_location="cpu")
        ev.update_batch(
            {"inputs": d["inputs"], "puzzle_identifiers": d["puzzle_identifiers"]},
            {"preds": d["preds"], "q_halt_logits": d["q_halt_logits"]},
        )
    tasks = []
    for name, puzzle in ev.test_puzzles.items():
        if name not in ev._local_preds:
            continue
        inv = V._build_invariants(puzzle["train"])
        for pair in puzzle["test"]:
            ih = grid_hash(arc_grid_to_np(pair["input"]))
            lh = grid_hash(arc_grid_to_np(pair["output"]))  # GOLD: used ONLY to label 'correct'
            preds = ev._local_preds[name].get(ih)
            if not preds:
                continue
            qmap = {}  # ph -> list of q_halt
            for ph, q in preds:
                qmap.setdefault(ph, []).append(float(q))
            til = _as_list(arc_grid_to_np(pair["input"]))
            app = V._applicability(inv, puzzle["train"], til)
            cands = []
            for ph, qs in qmap.items():
                try:
                    um, _md, feats = V._combined_scores(
                        _as_list(ev._local_hmap[ph]), inv, puzzle["train"], til, app
                    )
                    applied = [nm for nm in FEAT_NAMES if (nm == "v1" or app.get(nm, False))]
                    mindef = float(min(feats[FEAT_NAMES.index(nm)] for nm in applied))
                    umf = float(um)
                except Exception:
                    mindef, umf = 1e9, 1e9
                cands.append(
                    {
                        "votes": len(qs),
                        "q_mean": float(np.mean(qs)),
                        "q_max": float(np.max(qs)),
                        "q_lse": float(_logsumexp(qs)),
                        "min_defined": mindef,
                        "union_max": umf,
                        "correct": bool(ph == lh),
                    }
                )
            if cands:
                tasks.append({"task": name, "cands": cands})
    return tasks


def _pass(tasks, key, ks=(1, 2)):
    """pass@k fraction over tasks given a ranking key (sorts ascending = best first)."""
    hits = {k: 0 for k in ks}
    for t in tasks:
        ranked = sorted(t["cands"], key=key)
        rc = [c["correct"] for c in ranked]
        for k in ks:
            hits[k] += int(any(rc[:k]))
    n = len(tasks)
    return {f"pass@{k}": round(hits[k] / n, 4) for k in ks}


def run(write=True):
    started = time.time()
    tasks = _build_candidate_table()
    n = len(tasks)
    n_oracle = sum(1 for t in tasks if any(c["correct"] for c in t["cands"]))

    # ---- rankers (ascending key = best first; q_* are DESC-better so negate) ----
    rankers = {
        "TRM_VOTE": lambda c: (-c["votes"],),
        "Q_MEAN": lambda c: (-c["q_mean"], -c["votes"]),
        "Q_MAX": lambda c: (-c["q_max"], -c["votes"]),
        "Q_LSE_voteconfounded": lambda c: (-c["q_lse"], -c["votes"]),
        "HYBRID_vote_then_qmean": lambda c: (-c["votes"], -c["q_mean"]),
        "min_defined_handfeat": lambda c: (c["min_defined"], -c["votes"]),
        "union_max_handfeat": lambda c: (c["union_max"], -c["votes"]),
    }
    res = {name: _pass(tasks, key) for name, key in rankers.items()}

    # oracle ceiling
    oracle2 = round(n_oracle / n, 4)

    # ---- A0 vote-mimicry control ----
    # (a) within-task Spearman(q_mean, votes), averaged
    sp = [
        s
        for t in tasks
        if (s := _spearman([c["q_mean"] for c in t["cands"]], [c["votes"] for c in t["cands"]]))
        is not None
    ]
    spearman_qmean_vote = round(sum(sp) / len(sp), 4) if sp else None

    # (b) RESIDUAL ranker: global linear fit q_mean ~ a + b*vote, rank by residual (q_mean - pred)
    allc = [c for t in tasks for c in t["cands"]]
    vs = [c["votes"] for c in allc]
    qm = [c["q_mean"] for c in allc]
    vm, qmm = sum(vs) / len(vs), sum(qm) / len(qm)
    cov = sum((v - vm) * (q - qmm) for v, q in zip(vs, qm))
    var = sum((v - vm) ** 2 for v in vs) or 1.0
    b = cov / var
    a = qmm - b * vm
    for c in allc:
        c["_q_resid"] = c["q_mean"] - (a + b * c["votes"])
    res["Q_MEAN_residual_over_vote"] = _pass(tasks, lambda c: (-c["_q_resid"], -c["votes"]))

    # (c) discrimination AUROC(q_mean: gold vs non-gold). TWO forms:
    #   - WITHIN-TASK (selection-relevant): macro-average of per-task AUROC(gold vs non-gold). This is
    #     what matters -- selection ranks WITHIN a task. Used for the gate.
    #   - POOLED (between-task-inflated): gold-vs-nongold over ALL candidates. MISLEADING for selection
    #     because gold q_mean is globally high, so cross-task pairs (gold of A vs nongold of B) dominate
    #     and inflate it far above the within-task reality. Reported but NOT gated.
    g_all, ng_all = [], []
    per_task_auroc = []
    for t in tasks:
        gt = [c["q_mean"] for c in t["cands"] if c["correct"]]
        ngt = [c["q_mean"] for c in t["cands"] if not c["correct"]]
        g_all += gt
        ng_all += ngt
        a = _auroc(gt, ngt)
        if a is not None:
            per_task_auroc.append(a)
    qmean_auroc_within = (
        round(sum(per_task_auroc) / len(per_task_auroc), 4) if per_task_auroc else None
    )
    qmean_auroc_pooled = _auroc(g_all, ng_all)
    qmean_auroc_pooled = round(qmean_auroc_pooled, 4) if qmean_auroc_pooled is not None else None

    # ---- bootstrap-over-tasks 95% CI on pass@2(Q_MEAN) - pass@2(vote) ----
    # deterministic bootstrap (no Math.random): fixed resample indices via a simple LCG seeded at 0.
    def _lcg(seed):
        x = seed
        while True:
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            yield x

    gen = _lcg(12345)
    B = 1000
    deltas = []

    def _p2(sample_tasks, key):
        hits = sum(
            int(any(c["correct"] for c in sorted(t["cands"], key=key)[:2])) for t in sample_tasks
        )
        return hits / len(sample_tasks)

    kq = rankers["Q_MEAN"]
    kv = rankers["TRM_VOTE"]
    for _ in range(B):
        idx = [next(gen) % n for _ in range(n)]
        samp = [tasks[i] for i in idx]
        deltas.append(_p2(samp, kq) - _p2(samp, kv))
    deltas.sort()
    ci = [round(deltas[int(0.025 * B)], 4), round(deltas[int(0.975 * B)], 4)]
    delta_point = round(res["Q_MEAN"]["pass@2"] - res["TRM_VOTE"]["pass@2"], 4)

    # ---- gates ----
    # The Stage-0 QUESTION is: is q_halt a model-native PRIMARY selection signal beyond frequency? That
    # is answered by the DE-CONFOUNDED rankers ONLY (Q_MEAN as primary; the residual-over-vote ranker).
    # HYBRID (vote-primary, q_mean tie-break) and Q_LSE (logsumexp, vote-confounded by construction) are
    # NOT evidence for model-native signal -- a HYBRID win is a tie-break/deployment effect, a Q_LSE win
    # re-expresses vote. We report HYBRID separately as a deployment curiosity, not as the headline.
    vote2 = res["TRM_VOTE"]["pass@2"]
    deconf_rankers = ["Q_MEAN", "Q_MAX", "Q_MEAN_residual_over_vote"]
    best_deconf = max(deconf_rankers, key=lambda r: res[r]["pass@2"])
    sel_pass = res[best_deconf]["pass@2"] > vote2  # a DE-CONFOUNDED q ranker must beat vote
    resid_beats_vote = res["Q_MEAN_residual_over_vote"]["pass@2"] > vote2
    disc_pass = (qmean_auroc_within or 0) > 0.70  # WITHIN-task AUROC, not the inflated pooled one
    headroom = (res[best_deconf]["pass@2"] - vote2) / max(1e-9, oracle2 - vote2)
    # deployment-only side note: does q_halt as a tie-break help the deployable vote-primary ranker?
    hybrid_tiebreak_gain = round(res["HYBRID_vote_then_qmean"]["pass@2"] - vote2, 4)
    gates = {
        "selection_beats_vote": bool(sel_pass),
        "discrimination_within_task_auroc_gt_0p70": bool(disc_pass),
        "coverage_ge_0p80": True,  # q_halt defined for every candidate
        "headroom_capture_ge_0p30": bool(headroom >= 0.30),
        "headroom_capture_fraction": round(headroom, 4),
    }
    # model-native verdict: a de-confounded q ranker beats vote AND the residual (vote regressed out)
    # carries signal. Both must hold; the pooled AUROC is explicitly NOT trusted.
    model_native_real = bool(sel_pass and resid_beats_vote)

    verdict = (
        "complete: gap3_stage0_qhalt_"
        + (
            "model_native_signal_beats_vote_" + best_deconf
            if model_native_real
            else "scalar_qhalt_does_not_beat_vote_stage1_full_latent_required"
        )
        + f"_n{n}_vote_{vote2}_bestdeconf_{res[best_deconf]['pass@2']}"
        + f"_withinauroc_{qmean_auroc_within}_residbeatsvote_{resid_beats_vote}"
    )

    art = {
        "experiment": "arc3_gap3_stage0_qhalt_energy",
        "title": "GAP-3 Stage 0: TRM-native q_halt confidence as a selection energy vs frequency vote",
        "honest_verdict": verdict,
        "inference_substrate": "offline_trm_candidate_rerank_no_oracle",
        "n_tasks": n,
        "n_oracle_hit": n_oracle,
        "oracle_pass2_ceiling": oracle2,
        "rankers": res,
        "best_deconfounded_q_ranker": best_deconf,
        "best_deconfounded_q_ranker_beats_vote": bool(sel_pass),
        "best_deconfounded_q_ranker_note": (
            "'best' = highest pass@2 among the DE-CONFOUNDED q rankers; it does NOT mean it beats vote. "
            "Here best_deconf=Q_MAX TIES vote at 0.4516 with ZERO orthogonal lift (its hit set is "
            "byte-identical to vote's 14 hits — confirmed by the adversarial-verify round); it is parity, "
            "not a win. The headline (model_native_signal_real=False) is unaffected."
        ),
        "hybrid_tiebreak_gain_vs_vote_DEPLOYMENT_ONLY": hybrid_tiebreak_gain,
        "a0_vote_mimicry_control": {
            "within_task_spearman_qmean_vote": spearman_qmean_vote,
            "residual_ranker_pass2": res["Q_MEAN_residual_over_vote"]["pass@2"],
            "residual_beats_vote": bool(resid_beats_vote),
            "qmean_discrimination_auroc_within_task": qmean_auroc_within,
            "qmean_discrimination_auroc_pooled_INFLATED": qmean_auroc_pooled,
            "note": (
                "The POOLED AUROC is between-task-inflated (gold q_mean is globally high, so cross-task "
                "pairs dominate) and MUST NOT be read as selection power -- the WITHIN-task AUROC is the "
                "honest selection-relevant number. If residual_beats_vote is False AND within-task AUROC "
                "<= ~0.55, q_halt's apparent lift is collinear with vote-count (frequency in disguise), "
                "NOT a model-native signal. The residual ranker regresses q_mean on vote globally and "
                "ranks by the residual."
            ),
        },
        "model_native_signal_real": model_native_real,
        "stage0_synthesis": (
            "NEGATIVE for the model-native-PRIMARY hypothesis, but with an encouraging nuance for Stage "
            "1. The scalar q_halt has real SOFT within-task discrimination (gold beats most non-golds: "
            "within-task AUROC ~0.86) -- so TRM's confidence DOES carry correctness signal -- but the "
            "scalar is (a) not SHARP enough to seat gold top-2 against large candidate pools (Q_MEAN "
            "pass@2 0.29 < vote 0.45) and (b) largely REDUNDANT with frequency vote (residual-over-vote "
            "ranker collapses to 0.10; spearman(q_mean,vote) 0.44). Net: the lossy 1-D projection cannot "
            "beat vote. The 0.86 soft-AUROC is a LEADING INDICATOR that the full penultimate-activation "
            "latent (Stage 1) -- which is not collapsed to a scalar and not vote-redundant -- likely "
            "carries sharper, vote-orthogonal selection signal. GO for Stage 1 per the design's Stage-0 "
            "decision rule. (HYBRID vote-then-qmean gives +1 task, a marginal within-noise deployment "
            "nicety, NOT model-native evidence.)"
        ),
        "gates": gates,
        "bootstrap_pass2_delta_vs_vote": {
            "point": delta_point,
            "ci95_over_tasks": ci,
            "B": B,
            "note": "Bootstrap resamples TASKS with replacement (n tiny -> CI is wide; do not over-claim).",
        },
        "no_oracle_audit": (
            "Gold output read ONLY to LABEL correctness for scoring; NO ranker key uses it. q_halt is "
            "TRM's forward-pass halting logit (model-native), never derived from the solution."
        ),
        "honest_note": (
            "GAP-3 Stage 0 (operator: proceed). Cheapest model-native probe, zero GPU. A NEGATIVE here "
            "(scalar q_halt insufficient) does NOT kill GAP-3 -- it means the lossy scalar projects away "
            "the signal and Stage 1's full penultimate-activation latent is required. A POSITIVE green-"
            "lights Stage 1. Reported as-is per FALSE_NEGATIVE_RISK; positive control = oracle 0.61 > "
            "vote so selectable headroom is real. CAVEAT: n=31; re-confirm at 400-task scale."
        ),
        "no_gpu_used": True,
        "random_seed": 12345,
        "duration_s": round(time.time() - started, 1),
    }

    if write:
        Path(f"{CARNOT}/results/arc3_gap3_stage0_qhalt_energy.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n"
        )
        # compact candidate table for independent (no-torch) re-derivation by the adversarial workflow
        table = {
            "experiment": "arc3_gap3_stage0_candidate_table",
            "description": (
                "Per-(task,test) candidate rows from TRM's arc_v1 dump: votes, q_halt aggregations, "
                "hand-feature scores, correctness label. Lets the Stage-0 finding be re-derived in plain "
                "python WITHOUT the 270MB torch dump. correct uses gold ONLY for scoring."
            ),
            "n_tasks": n,
            "n_oracle_hit": n_oracle,
            "fields": ["votes", "q_mean", "q_max", "q_lse", "min_defined", "union_max", "correct"],
            "tasks": [
                {
                    "task": t["task"],
                    "cands": [
                        {
                            k: c[k]
                            for k in (
                                "votes",
                                "q_mean",
                                "q_max",
                                "q_lse",
                                "min_defined",
                                "union_max",
                                "correct",
                            )
                        }
                        for c in t["cands"]
                    ],
                }
                for t in tasks
            ],
        }
        Path(f"{CARNOT}/results/arc3_gap3_stage0_candidate_table.json").write_text(
            json.dumps(table, indent=2, sort_keys=True) + "\n"
        )

    print(f"-> {verdict}")
    print(f"   n_tasks={n} oracle_hit={n_oracle} oracle_pass2={oracle2}")
    for r in [
        "TRM_VOTE",
        "Q_MEAN",
        "Q_MAX",
        "Q_LSE_voteconfounded",
        "HYBRID_vote_then_qmean",
        "Q_MEAN_residual_over_vote",
        "min_defined_handfeat",
        "union_max_handfeat",
    ]:
        print(f"   {r:28s} pass@1={res[r]['pass@1']} pass@2={res[r]['pass@2']}")
    print(
        f"   A0: spearman(qmean,vote)={spearman_qmean_vote} resid_beats_vote={resid_beats_vote} "
        f"within_auroc={qmean_auroc_within} pooled_auroc(INFLATED)={qmean_auroc_pooled} "
        f"-> model_native_real={model_native_real}"
    )
    print(f"   HYBRID tie-break gain vs vote (deployment only): {hybrid_tiebreak_gain}")
    print(f"   gates={gates}")
    print(f"   bootstrap pass@2 delta vs vote: point={delta_point} CI95={ci}")
    return art


if __name__ == "__main__":
    run()
