"""DIAGNOSIS: WHY does the Carnot v2 hand-invariant ensemble ANTI-RANK on TRM's real candidate pool?

Context (results/arc3_trm_verifier_rerank.json, corrected 2026-06-09): with the verifier actually
scoring (after the numpy->list bug fix), the pure VERIFIER ranker scores pass@2 0.161 vs TRM_VOTE 0.484
vs oracle 0.613 on TRM's arc_v1 candidate pool. The hand-invariant ensemble (union_max over v1 + 7
families) is ANTI-correlated with correctness here. The operator asked to DIAGNOSE this before building
any new verifier: is the anti-rank a fixable aggregation artifact (one rogue family / max() too brittle)
or a genuine signal-absence (cheap hand-features just don't capture ARC rule-correctness)?

This script decomposes the verifier on TRM's REAL candidates (no synthetic distractors):

  1. PER-FAMILY discrimination AUROC: for each of {v1} + 7 families, over all (gold, non-gold) candidate
     pairs WITHIN a task, how often does the family give gold the LOWER (better) score? AUROC>0.5 =>
     the family ranks gold better (discriminative); ~0.5 => blind; <0.5 => ANTI-discriminative (it
     actively prefers wrong candidates). Threshold-free, tie-robust. Applicable-only per task.

  2. AGGREGATION pass@2 on TRM's pool: union_max (current) vs mean_defined vs min_defined vs median vs
     v1-only vs each LEAVE-ONE-FAMILY-OUT union_max. If dropping one family recovers ranking, that
     family is the culprit (fixable). If no aggregation beats ~vote, hand-features can't reach the
     headroom (=> GAP-3 learned/model-native energy is the only path).

  3. SINKING-FAMILY census: for gold candidates that union_max mis-ranks (a wrong candidate has lower
     union_max), which family is gold's argmax (the family driving gold's high violation)? Names the
     family that sinks correct answers.

NO oracle in the ranker, NO model re-run. Reuses the rerank's TRM de-aug replay + the v2 verifier's
_combined_scores feature vector (feats = [v1, symmetry, color_mapping, object_count, content_overlap,
delta_pattern, tiling_scaling, palette_histogram_shape]).

  ~/trm_venv/bin/python scripts/experiments/arc3_verifier_antirank_diagnosis.py
"""

from __future__ import annotations

import glob
import json
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

# feats index order from V._combined_scores: [v1] + V.FAMILY_ORDER
FEAT_NAMES = ["v1", *V.FAMILY_ORDER]


def _as_list(g):
    return g.tolist() if isinstance(g, np.ndarray) else g


def _auroc(pos, neg):
    """P(random pos score > random neg score), ties=0.5. Here we feed -score so that LOWER violation
    (better) => HIGHER fed value, and AUROC>0.5 means gold scored lower/better than non-gold."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def _pass2(cands, key):
    """pass@2 over one task's candidates given a sort key (ascending = better). Tie-break by votes desc
    so a non-discriminative key falls back to vote, not arbitrary order (isolates the key's lift)."""
    ranked = sorted(cands, key=lambda c: (key(c), -c["votes"]))
    return any(c["correct"] for c in ranked[:2])


def run(write=True):
    started = time.time()
    ev = ARC(data_path=DATA, eval_metadata=SimpleNamespace(blank_identifier_id=0))
    for sh in sorted(glob.glob(PREDS_GLOB)):
        d = torch.load(sh, map_location="cpu")
        ev.update_batch(
            {"inputs": d["inputs"], "puzzle_identifiers": d["puzzle_identifiers"]},
            {"preds": d["preds"], "q_halt_logits": d["q_halt_logits"]},
        )

    # per-family (gold vs non-gold) score lists, pooled across tasks for a global discrimination AUROC
    fam_gold = {nm: [] for nm in FEAT_NAMES}  # gold candidate family scores
    fam_nongold = {nm: [] for nm in FEAT_NAMES}  # non-gold candidate family scores
    fam_applicable_tasks = {nm: 0 for nm in FEAT_NAMES}

    # aggregation pass@2 accumulators
    aggs = ["union_max", "mean_defined", "min_defined", "median_defined", "v1_only"]
    loo = [f"loo_drop_{nm}" for nm in V.FAMILY_ORDER]  # leave-one-family-out union_max
    agg_hits = {a: 0 for a in [*aggs, *loo, "TRM_VOTE"]}

    sinking = {
        nm: 0 for nm in FEAT_NAMES
    }  # which family is gold's union_max argmax when mis-ranked
    n_tasks = 0
    n_oracle = 0
    n_misranked = 0

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
            ti = arc_grid_to_np(pair["input"])
            til = _as_list(ti)
            app = V._applicability(inv, puzzle["train"], til)
            # applicable family set for THIS task (for the defined-aggregations + applicable-only AUROC)
            applicable = {nm: (nm == "v1" or app.get(nm, False)) for nm in FEAT_NAMES}

            cands = []
            for ph, (cnt, sq) in cand.items():
                grid = ev._local_hmap[ph]
                try:
                    um, md, feats = V._combined_scores(
                        _as_list(grid), inv, puzzle["train"], til, app
                    )
                except Exception:
                    continue
                fv = {nm: float(feats[i]) for i, nm in enumerate(FEAT_NAMES)}
                cands.append(
                    {
                        "ph": ph,
                        "votes": cnt,
                        "feats": fv,
                        "um": float(um),
                        "md": float(md),
                        "correct": (ph == lh),
                    }
                )
            if not cands:
                continue
            n_tasks += 1
            has_gold = any(c["correct"] for c in cands)
            if has_gold:
                n_oracle += 1

            # ---- per-family pooled gold-vs-nongold scores (applicable-only) ----
            for nm in FEAT_NAMES:
                if not applicable[nm]:
                    continue
                fam_applicable_tasks[nm] += 1
                for c in cands:
                    (fam_gold if c["correct"] else fam_nongold)[nm].append(c["feats"][nm])

            # ---- aggregation keys (defined = applicable families only) ----
            applied = [nm for nm in FEAT_NAMES if applicable[nm]]

            def _defined_vals(c, drop=None):
                return [c["feats"][nm] for nm in applied if nm != drop]

            keymap = {
                "union_max": lambda c: max(_defined_vals(c)),
                "mean_defined": lambda c: sum(_defined_vals(c)) / len(_defined_vals(c)),
                "min_defined": lambda c: min(_defined_vals(c)),
                "median_defined": lambda c: float(np.median(_defined_vals(c))),
                "v1_only": lambda c: c["feats"]["v1"],
                "TRM_VOTE": lambda c: -c["votes"],
            }
            for fam in V.FAMILY_ORDER:
                keymap[f"loo_drop_{fam}"] = lambda c, _f=fam: max(_defined_vals(c, drop=_f))

            for a, key in keymap.items():
                if _pass2(cands, key):
                    agg_hits[a] += 1

            # ---- sinking-family census on union_max mis-ranks ----
            if has_gold:
                gold = next(c for c in cands if c["correct"])
                sel = min(cands, key=lambda c: c["um"])
                if not sel["correct"] and sel["um"] < gold["um"]:
                    n_misranked += 1
                    # which applicable family is gold's argmax (drives its high union_max)?
                    arg = max(applied, key=lambda nm: gold["feats"][nm])
                    sinking[arg] += 1

    def rate(x):
        return round(x / n_tasks, 4) if n_tasks else None

    # global per-family discrimination AUROC (feed -score so >0.5 = gold better)
    fam_auroc = {}
    for nm in FEAT_NAMES:
        a = _auroc([-x for x in fam_gold[nm]], [-x for x in fam_nongold[nm]])
        fam_auroc[nm] = round(a, 4) if a is not None else None

    agg_pass2 = {a: rate(h) for a, h in agg_hits.items()}
    best_agg = max([*aggs, *loo], key=lambda a: agg_hits[a])

    # honest verdict: does ANY hand-feature aggregation beat TRM_VOTE?
    beats_vote = [a for a in [*aggs, *loo] if (agg_pass2[a] or 0) > (agg_pass2["TRM_VOTE"] or 0)]
    verdict = (
        "complete: antirank_diagnosis_"
        + (
            "hand_features_salvageable_" + best_agg
            if beats_vote
            else "no_hand_aggregation_beats_vote_signal_absent"
        )
        + f"_n{n_tasks}_bestagg_{agg_pass2[best_agg]}_vote_{agg_pass2['TRM_VOTE']}"
    )

    art = {
        "experiment": "arc3_verifier_antirank_diagnosis",
        "title": "Why the Carnot v2 hand-invariant ensemble anti-ranks on TRM's real candidate pool",
        "honest_verdict": verdict,
        "inference_substrate": "offline_trm_candidate_rerank_no_oracle",
        "n_tasks": n_tasks,
        "n_oracle_hit": n_oracle,
        "n_union_max_misranked": n_misranked,
        "per_family_discrimination_auroc": fam_auroc,
        "per_family_auroc_note": (
            "AUROC over all (gold, non-gold) candidate pairs within a task, applicable-only, fed -score "
            "so >0.5 = family scores gold LOWER (better/discriminative), ~0.5 = blind, <0.5 = "
            "ANTI-discriminative (prefers wrong candidates). This is on TRM's REAL mis-predictions, NOT "
            "the synthetic distractor protocol the v2 verifier was tuned on."
        ),
        "family_applicable_task_counts": fam_applicable_tasks,
        "aggregation_pass2": agg_pass2,
        "best_aggregation": best_agg,
        "aggregations_beating_vote": beats_vote,
        "sinking_family_census": sinking,
        "sinking_family_note": (
            "Of tasks where union_max ranks a wrong candidate above gold, which applicable family is "
            "gold's argmax (drives gold's high max-violation). Names the family sinking correct answers."
        ),
        "honest_note": (
            "DIAGNOSIS per operator directive (2026-06-09). If aggregations_beating_vote is empty AND "
            "every per_family AUROC <= ~0.55, the cheap hand-features genuinely lack ARC rule-correctness "
            "signal on real candidates => GAP-3 learned/model-native energy is the only path (not more "
            "invariants). If one leave-one-out recovers ranking, that family is a fixable rogue."
        ),
        "no_gpu_used": True,
        "random_seed": 0,
        "duration_s": round(time.time() - started, 1),
    }
    if write:
        Path(f"{CARNOT}/results/arc3_verifier_antirank_diagnosis.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n"
        )

    print(f"-> {verdict}")
    print(f"   n_tasks={n_tasks} oracle_hit={n_oracle} union_max_misranked={n_misranked}")
    print("   per-family discrimination AUROC (>0.5 good, <0.5 ANTI):")
    for nm in FEAT_NAMES:
        print(f"     {nm:24s} {fam_auroc[nm]}  (applicable in {fam_applicable_tasks[nm]} tasks)")
    print("   aggregation pass@2 (TRM_VOTE = baseline):")
    for a in ["TRM_VOTE", *aggs, *loo]:
        flag = "  <-- beats vote" if a in beats_vote else ""
        print(f"     {a:24s} {agg_pass2[a]}{flag}")
    print(f"   sinking-family census (union_max mis-ranks): {sinking}")
    return art


if __name__ == "__main__":
    run()
