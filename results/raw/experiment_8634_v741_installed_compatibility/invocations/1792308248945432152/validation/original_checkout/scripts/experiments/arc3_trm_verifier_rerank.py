"""TRM step 2: does the Carnot energy/invariant verifier RERANK TRM's candidate grids better than its
own augmentation-frequency vote? (The verifier-as-asset test on a SOTA generator + known benchmark.)

Plan: results/trm_verifier_rerank_opportunity.json — TRM's correct answer is present-but-mis-voted on
~10pp of tasks (pass@1000 - pass@2). This experiment tests whether the Carnot ARC invariant ensemble
(arc_grid_verifier_invariants_v2_combined: structural v1 + train-pair transformation-consistency
families) can SELECT the correct candidate where TRM's frequency vote mis-picks it.

Method (NO oracle, NO re-running the model):
  1. Reuse the TRM ARC evaluator's EXACT de-aug + voting accumulation (update_batch) on the dumped
     per-augmentation predictions (eval_out/arc_v1/step_0_all_preds.*; produced by the harness with
     --save_outputs incl. q_halt_logits). This yields, per task, the unique candidate grids + their
     vote_count + avg_q + which one is correct.
  2. For each task, build invariants from its TRAIN pairs (test_puzzles.json[task]['train']) and score
     every candidate with the verifier (union_max violation; lower = more consistent).
  3. Compare three rankers on pass@1 / pass@2, vs the oracle pass@K ceiling:
       - TRM_VOTE   : sort by [vote_count, avg_q] desc (the published baseline)
       - VERIFIER   : sort by ascending union_max (then mean_defined)        (pure verifier)
       - HYBRID     : TRM vote PRIMARY, verifier as structural-prune + within-top-cluster tie-break
                      (the survey's lowest-honest-negative-exposure framing)
  4. Report flips (top-1 changed vs TRM_VOTE), net fixes, and whether any ranker raises pass@2.

Honest framing (per the design survey): TRM's mis-votes are often structurally valid, so a
structural-only rerank is expected to be a wash; the transformation-consistency families are the
real lever, but the honest-negative risk is concrete (transpose-invariant + variable-output-dim tasks
sit in the verifier's null space). This experiment is the POSITIVE CONTROL on TRM's REAL candidate
pool — report whatever it shows, including a wash/negative.

  ~/trm_venv/bin/python scripts/experiments/arc3_trm_verifier_rerank.py
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
from evaluators.arc import ARC  # noqa: E402  (reuse the published de-aug/voting)
from dataset.build_arc_dataset import arc_grid_to_np, grid_hash  # noqa: E402
import arc_grid_verifier_invariants_v2_combined as V  # noqa: E402,N812  (Carnot rerank ensemble)

import os  # noqa: E402

GAP1 = (
    os.environ.get("CARNOT_GAP1") == "1"
)  # fold the directional-adjacency (GAP-1) family into union_max
if GAP1:
    import arc_invariant_directional_adjacency_draft as DA  # noqa: E402,N812


def _as_list(g):
    """V._combined_scores' grid helpers (_dims/_colors/_palette_counts) require list[list[int]] and
    raise `ValueError: truth value of an array ... ambiguous` on a numpy grid (the `len(g[0]) if g`
    idiom). TRM's _local_hmap candidates AND arc_grid_to_np(test_input) are numpy, so they MUST be
    converted before scoring — passing numpy silently routed EVERY candidate to the 1e9 except branch,
    leaving the verifier degenerate (all-tied) for the whole rerank. DA.family_score np.asarray()s
    internally, so it is type-agnostic."""
    return g.tolist() if isinstance(g, np.ndarray) else g


def _verifier_scores(cand_grid, inv, train_pairs, test_input, app):
    """(union_max, mean_defined, da) for one candidate; lower = fewer violations. `da` is the GAP-1
    directional-adjacency (orientation) violation — kept SEPARATE (a tie-break key), NOT folded into
    union_max's max (where it gets buried), so it can decide between a grid and its transpose."""
    try:
        um, md, _ = V._combined_scores(
            _as_list(cand_grid), inv, train_pairs, _as_list(test_input), app
        )
        da = float(DA.family_score(cand_grid, inv, train_pairs, test_input)) if GAP1 else 0.0
        return float(um), float(md), da
    except Exception:
        return 1e9, 1e9, 1e9  # un-scorable -> treat as maximally inconsistent (pruned)


def _pass_at_k(ranked_correct, ks=(1, 2)):
    """ranked_correct: list of bool (is_correct) in rank order. Returns {k: any(top-k correct)}."""
    return {k: bool(any(ranked_correct[:k])) for k in ks}


def run(write=True):
    started = time.time()
    meta = SimpleNamespace(blank_identifier_id=0)
    ev = ARC(data_path=DATA, eval_metadata=meta)

    # --- replay the EXACT de-aug + accumulation over the dumped predictions ---
    shards = sorted(glob.glob(PREDS_GLOB))
    if not shards:
        art = {
            "experiment": "arc3_trm_verifier_rerank",
            "honest_verdict": "blocked_no_dumped_preds",
            "inference_substrate": "offline_trm_candidate_rerank",
            "note": f"no preds at {PREDS_GLOB}",
        }
        if write:
            Path(f"{CARNOT}/results/arc3_trm_verifier_rerank.json").write_text(
                json.dumps(art, indent=2) + "\n"
            )
        print("-> blocked_no_dumped_preds")
        return art
    for sh in shards:
        d = torch.load(sh, map_location="cpu")
        ev.update_batch(
            {"inputs": d["inputs"], "puzzle_identifiers": d["puzzle_identifiers"]},
            {"preds": d["preds"], "q_halt_logits": d["q_halt_logits"]},
        )

    # --- per task: build candidates, score, rank three ways ---
    rankers = ["TRM_VOTE", "VERIFIER", "HYBRID"]
    agg = {r: {1: 0, 2: 0} for r in rankers}
    oracle = {k: 0 for k in (1, 2, 5, 10, 100, 1000)}
    n_tasks = 0
    flips = {"VERIFIER": 0, "HYBRID": 0}
    net_fix = {"VERIFIER": 0, "HYBRID": 0}  # +1 fixed a wrong vote, -1 broke a right vote
    per_task = []
    # missing-verifier gap census (ops/verifier_gaps.md): when the correct answer is IN the candidate
    # pool (oracle hit) but HYBRID's top-2 misses it, bucket WHY the verifier couldn't select it.
    gap_buckets = {
        "GAP1_transpose_orientation": 0,
        "GAP2_variable_dim_abstention": 0,
        "GAP3_content_in_null_space": 0,
        "uncaptured_total": 0,
    }

    def _dihedral_equiv(a, b):
        a = np.asarray(a)
        b = np.asarray(b)
        for op in (
            a,
            a.T,
            np.rot90(a),
            np.rot90(a, 2),
            np.rot90(a, 3),
            np.fliplr(a),
            np.flipud(a),
            np.fliplr(a.T),
        ):
            if op.shape == b.shape and np.array_equal(op, b):
                return True
        return False

    def _variable_output_dim(train):
        # positional/transform families need a fixed output canvas; if the train OUTPUT shapes vary,
        # they abstain (GAP-2). Proxy: train outputs are not all the same shape.
        return len({np.asarray(p["output"]).shape for p in train}) > 1

    for name, puzzle in ev.test_puzzles.items():
        if name not in ev._local_preds:
            continue  # task not in the dumped (capped) subset
        inv = V._build_invariants(puzzle["train"])
        for pair in puzzle["test"]:
            ih = grid_hash(arc_grid_to_np(pair["input"]))
            lh = grid_hash(arc_grid_to_np(pair["output"]))
            preds = ev._local_preds[name].get(ih)
            if not preds:
                continue
            # group augmented predictions -> unique candidates with vote_count + avg_q
            cand = {}
            for ph, q in preds:
                c = cand.setdefault(ph, [0, 0.0])
                c[0] += 1
                c[1] += q
            cands = []
            test_input = arc_grid_to_np(pair["input"])
            app = V._applicability(inv, puzzle["train"], _as_list(test_input))
            for ph, (cnt, sq) in cand.items():
                grid = ev._local_hmap[ph]
                um, md, da = _verifier_scores(grid, inv, puzzle["train"], test_input, app)
                cands.append(
                    {
                        "ph": ph,
                        "votes": cnt,
                        "avg_q": sq / cnt,
                        "um": um,
                        "md": md,
                        "da": da,
                        "correct": (ph == lh),
                    }
                )
            if not cands:
                continue
            n_tasks += 1
            for k in oracle:
                oracle[k] += int(
                    any(c["correct"] for c in cands)
                )  # any candidate correct (ceiling, K>=n)

            # TRM_VOTE: votes desc, avg_q desc
            r_vote = sorted(cands, key=lambda c: (c["votes"], c["avg_q"]), reverse=True)
            # VERIFIER: union_max asc, mean_defined asc
            r_ver = sorted(cands, key=lambda c: (c["um"], c["md"]))
            # HYBRID: vote PRIMARY; within the top-vote cluster (votes within 80% of max) the verifier
            # breaks ties. ORDER MATTERS: `da` (GAP-1 directional-adjacency / orientation) leads `um`
            # because um (object_count + palette_histogram) is PROVABLY transpose-invariant, so it cannot
            # separate a grid from its transpose — only `da` can. When GAP1 is off, da==0.0 for every
            # candidate (a constant), so the key collapses to the v2 baseline ordering (cluster, um,
            # -votes, md) and HYBRID is byte-identical to the committed baseline.
            maxv = max(c["votes"] for c in cands)
            r_hyb = sorted(
                cands,
                key=lambda c: (-(c["votes"] >= 0.8 * maxv), c["da"], c["um"], -c["votes"], c["md"]),
            )
            ranked = {"TRM_VOTE": r_vote, "VERIFIER": r_ver, "HYBRID": r_hyb}
            base_top1 = r_vote[0]["correct"]
            row = {"task": name, "n_candidates": len(cands), "base_top1_correct": base_top1}
            for r in rankers:
                rc = [c["correct"] for c in ranked[r]]
                p = _pass_at_k(rc)
                agg[r][1] += int(p[1])
                agg[r][2] += int(p[2])
                row[f"{r}_pass@1"] = p[1]
                row[f"{r}_pass@2"] = p[2]
            for r in ("VERIFIER", "HYBRID"):
                top1 = ranked[r][0]["correct"]
                if ranked[r][0]["ph"] != r_vote[0]["ph"]:
                    flips[r] += 1
                    if top1 and not base_top1:
                        net_fix[r] += 1
                    elif base_top1 and not top1:
                        net_fix[r] -= 1
            # MISSING-VERIFIER GAP CENSUS: correct answer IS in the pool but HYBRID top-2 missed it ->
            # which missing discriminator would have been needed? (feeds ops/verifier_gaps.md)
            correct_cands = [c for c in cands if c["correct"]]
            if correct_cands and not any(c["correct"] for c in r_hyb[:2]):
                gap_buckets["uncaptured_total"] += 1
                cc_grid = ev._local_hmap[correct_cands[0]["ph"]]
                sel_grid = ev._local_hmap[r_hyb[0]["ph"]]
                if _dihedral_equiv(cc_grid, sel_grid):
                    gap_buckets["GAP1_transpose_orientation"] += 1  # GAP-1
                elif _variable_output_dim(puzzle["train"]):
                    gap_buckets["GAP2_variable_dim_abstention"] += 1  # GAP-2
                else:
                    gap_buckets["GAP3_content_in_null_space"] += 1  # GAP-3
            per_task.append(row)

    def rate(x):
        return round(x / n_tasks, 4) if n_tasks else None

    res = {r: {"pass@1": rate(agg[r][1]), "pass@2": rate(agg[r][2])} for r in rankers}
    oracle_rates = {f"pass@{k}": rate(v) for k, v in oracle.items()}
    best = max(rankers, key=lambda r: agg[r][2])
    captured = (res[best]["pass@2"] or 0) - (res["TRM_VOTE"]["pass@2"] or 0)
    verdict = (
        f"complete: trm_verifier_rerank_n{n_tasks}_trmvote_pass2_{res['TRM_VOTE']['pass@2']}"
        f"_best_{best}_pass2_{res[best]['pass@2']}_captured_{round(captured, 4)}"
        f"_oracle_{oracle_rates['pass@1000']}"
    )
    art = {
        "experiment": "arc3_trm_verifier_rerank",
        "title": "TRM candidate rerank by the Carnot invariant verifier",
        "honest_verdict": verdict,
        "inference_substrate": "offline_trm_candidate_rerank_no_oracle",
        "n_tasks": n_tasks,
        "rankers": res,
        "oracle_ceiling": oracle_rates,
        "trm_vote_pass2": res["TRM_VOTE"]["pass@2"],
        "best_ranker": best,
        "best_ranker_pass2": res[best]["pass@2"],
        "gap_captured_pass2_vs_trmvote": round(captured, 4),
        "present_but_misvoted_headroom": round(
            (oracle_rates["pass@1000"] or 0) - (res["TRM_VOTE"]["pass@2"] or 0), 4
        ),
        "flips_top1_vs_vote": flips,
        "net_fixes": net_fix,
        "missing_verifier_gaps": gap_buckets,
        "missing_verifier_gaps_note": (
            "Of the tasks where the correct answer is in TRM's candidate pool "
            "but the HYBRID verifier-rerank still missed it, this buckets WHY "
            "(maps to ops/verifier_gaps.md GAP-1 transpose/orientation, GAP-2 "
            "variable-output-dim abstention, GAP-3 content-in-null-space). Each "
            "non-zero bucket is empirical evidence to build the missing verifier "
            "— the project's core product backlog (CLAUDE.md Missing-Verifier "
            "Gap Logging)."
        ),
        "framing": (
            "HYBRID = TRM vote primary + verifier as structural-prune/within-top-cluster tie-break "
            "(lowest honest-negative exposure per the design survey)."
        ),
        "honest_note": (
            "POSITIVE CONTROL on TRM's REAL candidate pool. A wash/negative is reported as-is "
            "(per FALSE_NEGATIVE_RISK discipline): structural-only rerank is expected to be a "
            "wash; the lever is the train-pair transformation-consistency families, bounded by "
            "the transpose-invariant + variable-output-dim null space."
        ),
        "n_complete_tasks_caveat": "subset of ~29 dumped tasks (small sample); the gap is directional.",
        "no_gpu_used": True,
        "submitted_to_leaderboard": False,
        "random_seed": 0,
        "duration_s": round(time.time() - started, 1),
        "per_task": per_task[:60],
    }
    art["gap1_directional_adjacency_folded_in"] = GAP1
    if write:
        suffix = "_gap1" if GAP1 else ""
        Path(f"{CARNOT}/results/arc3_trm_verifier_rerank{suffix}.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n"
        )
    print(f"-> {verdict}")
    print(f"   n_tasks={n_tasks}  oracle_ceiling(pass@1000)={oracle_rates['pass@1000']}")
    for r in rankers:
        print(f"   {r:9} pass@1={res[r]['pass@1']}  pass@2={res[r]['pass@2']}")
    print(
        f"   flips vs vote: {flips} | net fixes: {net_fix} | captured(best-vote, pass@2): {round(captured, 4)}"
    )
    return art


if __name__ == "__main__":
    run()
