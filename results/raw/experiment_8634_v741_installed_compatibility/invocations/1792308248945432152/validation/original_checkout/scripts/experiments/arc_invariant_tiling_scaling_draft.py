"""DRAFT (north-star domain): a CHEAP, NON-INDUCING tiling/scaling invariant that catches
CONTENT errors on the SCALE-RULE subset of ARC-AGI-1.

WHY (north-star.md §0 + the v1 ceiling). The v1 grid verifier
(results/arc_grid_verifier_discriminator.json) PRUNES structurally-malformed candidates
well (wrong_task_gold 0.947, random 0.889, wrong_dim 0.863) but is BLIND to CONTENT errors
that preserve the dimension/palette/background invariants: perturbed_gold 0.006,
transposed 0.293, color_swap 0.384, blank 0.135, copy_input 0.389. Those are the ceiling.

This invariant attacks the ceiling on ONE structured family: scale-rule tasks, where the
output dims = (ky, kx) * input dims. On that subset the output is almost always a STRUCTURED
TILING / UPSCALING of the input:

  * subtile-transform model: the output partitions into ky x kx subtiles, each the size of
    the input, and each subtile is either (a) a member of a SMALL transform vocabulary of
    the test input {identity, fliplr, flipud, flip180, transpose, anti-transpose, rot90,
    rot270} or (b) a constant (single-colour) fill. 16/24 ARC-1 scale tasks fit this.
  * block-upscale model: the output partitions into ih x iw blocks of size ky x kx, each
    block CONSTANT (one colour). 4/24 fit this (it is the classic "each input cell -> a
    kxk solid block" upscale).

The scorer picks whichever model BEST EXPLAINS the train-pair OUTPUTS (lowest mean
violation across the train pairs -- NO test-gold used, no label leak), then scores a
candidate by how much it VIOLATES that model's decomposition relative to the TEST INPUT.

  family_score(cand, inv, train_pairs, test_input) -> float   # LOWER = more consistent

Crucially this is CHEAP and DETERMINISTIC: no rule induction by search, no LLM, no GPU. It
is a structural/relational check derivable from the train pairs alone. The transform
vocabulary is FIXED (8 rigid symmetries) -- we do NOT search for an arbitrary rule, we just
verify the candidate is a valid tiling/upscaling of the *test input*. A perturbed gold
(one cell wrong inside an otherwise-clean subtile) breaks subtile equality and scores high.

HONEST scope: this invariant is meaningful ONLY on the scale-rule subset. On non-scale
tasks it ABSTAINS (returns 0.0, a neutral score) so it neither helps nor hurts. The headline
question is therefore: on the SCALE subset's test inputs, does it lift gold above the HARD
content-preserving distractors (perturbed_gold especially) past the v1 ceiling?

  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    .venv/bin/python scripts/experiments/arc_invariant_tiling_scaling_draft.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

from arc_grid_verifier_discriminator_draft import (  # noqa: E402  (reuse v1 — pure, no heavy deps)
    _auroc,
    _bg,
    _build_invariants,
    _colors,
    _dims,
    _distractors,
    _predicted_dims,
)

ARC = Path("/home/ianblenke/trm_src/kaggle/combined")
OUT = REPO_ROOT / "results" / "experiment_arc_invariant_tiling_scaling.json"


# ---------------------------------------------------------------------------
# Rigid-symmetry transform vocabulary (FIXED — NOT a search space). These are the
# 8 dihedral symmetries; the non-square ones (transpose/rot90/...) only apply when h==w
# because otherwise they would change the grid shape and could not equal a same-shaped subtile.
# ---------------------------------------------------------------------------
def _transform_set(g):
    """Return the set (as a list) of rigid-symmetry transforms of grid g that preserve its shape."""
    h, w = _dims(g)
    outs = [
        g,                                   # identity
        [row[::-1] for row in g],            # fliplr
        g[::-1],                             # flipud
        [row[::-1] for row in g[::-1]],      # flip180
    ]
    if h == w and h > 0:
        outs.append([[g[r][c] for r in range(h)] for c in range(w)])              # transpose
        outs.append([[g[h - 1 - r][w - 1 - c] for r in range(h)] for c in range(w)])  # anti-transpose
        outs.append([[g[h - 1 - r][c] for r in range(h)] for c in range(w)])      # rot90 (ccw)
        outs.append([[g[r][w - 1 - c] for r in range(h)] for c in range(w)])      # rot270
    # de-dup (e.g. symmetric grids collapse transforms)
    uniq = []
    for o in outs:
        if o not in uniq:
            uniq.append(o)
    return uniq


def _subtile(grid, i, j, ih, iw):
    """Extract the (i,j)-th subtile of size (ih,iw) from grid (which must be ky*ih by kx*iw)."""
    return [[grid[i * ih + r][j * iw + c] for c in range(iw)] for r in range(ih)]


def _is_constant(tile):
    return len({x for row in tile for x in row}) <= 1


# ---------------------------------------------------------------------------
# Violation under the SUBTILE-TRANSFORM model.
# The output partitions into ky x kx subtiles each the size of `src`; a subtile is "clean"
# if it equals some rigid-symmetry transform of `src` OR is a single-colour constant fill
# (the common "blank / solid quadrant" case in fractal tilings). Violation = fraction of
# subtiles that are neither. LOWER is better.
# ---------------------------------------------------------------------------
def _subtile_violation(src, cand, ky, kx):
    ih, iw = _dims(src)
    ch, cw = _dims(cand)
    if ih == 0 or iw == 0 or ch != ky * ih or cw != kx * iw:
        return 1.0  # candidate dims are not a clean (ky,kx) multiple of the input — max violation
    tset = _transform_set(src)
    bad = 0
    total = ky * kx
    for i in range(ky):
        for j in range(kx):
            st = _subtile(cand, i, j, ih, iw)
            if _is_constant(st):
                continue
            if not any(st == tg for tg in tset):
                bad += 1
    return bad / total if total else 1.0


# ---------------------------------------------------------------------------
# Violation under the BLOCK-UPSCALE model.
# The output partitions into ih x iw blocks each of size (ky,kx); a block is "clean" if it
# is a single colour (each input cell expanded to a solid kxk block). Violation = fraction
# of non-constant blocks. LOWER is better.
# ---------------------------------------------------------------------------
def _block_violation(src, cand, ky, kx):
    ih, iw = _dims(src)
    ch, cw = _dims(cand)
    if ih == 0 or iw == 0 or ch != ky * ih or cw != kx * iw:
        return 1.0
    bad = 0
    total = ih * iw
    for r in range(ih):
        for c in range(iw):
            block = {cand[r * ky + dy][c * kx + dx] for dy in range(ky) for dx in range(kx)}
            if len(block) > 1:
                bad += 1
    return bad / total if total else 1.0


# ---------------------------------------------------------------------------
# Model selection from the TRAIN PAIRS ONLY (no test gold). We compute each model's mean
# violation on the train-pair OUTPUTS (using each train pair's own input as src) and pick
# the model with the lower mean. This is the "best-explaining structural model" — purely a
# function of the train pairs, evaluated once per task.
# ---------------------------------------------------------------------------
def _select_model(train, ky, kx):
    sub = [_subtile_violation(p["input"], p["output"], ky, kx) for p in train]
    blk = [_block_violation(p["input"], p["output"], ky, kx) for p in train]
    mean_sub = sum(sub) / len(sub) if sub else 1.0
    mean_blk = sum(blk) / len(blk) if blk else 1.0
    # Prefer subtile-transform on ties (it is the more discriminating / content-sensitive model).
    if mean_blk < mean_sub:
        return "block", mean_blk
    return "subtile", mean_sub


# ---------------------------------------------------------------------------
# THE INVARIANT-FAMILY SCORER.
#   family_score(cand, inv, train_pairs, test_input) -> float   (LOWER = more consistent)
# Abstains (0.0) on non-scale tasks. On scale tasks: scores the candidate's violation under
# the train-selected structural model, with a dimension-gate so a candidate that does NOT
# have the predicted (scaled) dims is maximally penalised (it cannot be a valid tiling).
# ---------------------------------------------------------------------------
def family_score(cand, inv, train_pairs, test_input):
    kind, val = inv["dim_rule"]
    if kind != "scale" or val is None:
        return 0.0  # ABSTAIN — this family only speaks to scale-rule tasks
    ky, kx = val
    pred = _predicted_dims(inv, test_input)
    if pred is not None and _dims(cand) != pred:
        return 1.0  # wrong dims -> cannot be a valid (ky,kx) tiling of the test input
    model, _train_fit = _select_model(train_pairs, ky, kx)
    if model == "block":
        return _block_violation(test_input, cand, ky, kx)
    return _subtile_violation(test_input, cand, ky, kx)


# ---------------------------------------------------------------------------
# Measurement on the SAME protocol as v1.
# ---------------------------------------------------------------------------
def run(split="training", limit=None, seed=0, write=True):
    rng = random.Random(seed)
    ch = json.load(open(ARC / f"arc-agi_{split}_challenges.json"))
    so = json.load(open(ARC / f"arc-agi_{split}_solutions.json"))
    task_ids = list(ch)
    if limit:
        task_ids = task_ids[:limit]
    all_golds = [so[t][0] for t in task_ids if so.get(t)]

    # per-distractor pairs across ALL tasks (for the overall, apples-to-apples-with-v1 numbers)
    per_distr = {}            # kind -> list[(gold_score, distractor_score)]
    # per-distractor pairs on the SCALE SUBSET ONLY (where this family actually fires)
    per_distr_scale = {}
    gold_e, easy_e, hard_e = [], [], []
    gold_e_s, easy_e_s, hard_e_s = [], [], []
    n_eval = 0
    n_eval_scale = 0
    n_scale_tasks = 0

    for t in task_ids:
        task = ch[t]
        inv = _build_invariants(task["train"])
        is_scale = inv["dim_rule"][0] == "scale"
        if is_scale:
            n_scale_tasks += 1
        for ti, test in enumerate(task["test"]):
            if not so.get(t) or ti >= len(so[t]):
                continue
            gold = so[t][ti]
            tin = test["input"]
            easy, hard = _distractors(gold, tin, all_golds, rng)
            if not easy and not hard:
                continue
            n_eval += 1
            gs = family_score(gold, inv, task["train"], tin)
            gold_e.append(-gs)
            for kind, d in list(easy.items()) + list(hard.items()):
                ds = family_score(d, inv, task["train"], tin)
                per_distr.setdefault(kind, []).append((gs, ds))
            easy_e += [-family_score(d, inv, task["train"], tin) for d in easy.values()]
            hard_e += [-family_score(d, inv, task["train"], tin) for d in hard.values()]

            if is_scale:
                n_eval_scale += 1
                gold_e_s.append(-gs)
                for kind, d in list(easy.items()) + list(hard.items()):
                    ds = family_score(d, inv, task["train"], tin)
                    per_distr_scale.setdefault(kind, []).append((gs, ds))
                easy_e_s += [-family_score(d, inv, task["train"], tin) for d in easy.values()]
                hard_e_s += [-family_score(d, inv, task["train"], tin) for d in hard.values()]

    def _summarize(per):
        out = {}
        for kind, pairs in sorted(per.items()):
            wins = sum(1 for ge, de in pairs if ge < de)
            ties = sum(1 for ge, de in pairs if ge == de)
            # pairwise AUROC: -score as the ranking signal (gold should rank ABOVE distractor)
            auroc = _auroc([-ge for ge, _ in pairs], [-de for _, de in pairs])
            out[kind] = {
                "n": len(pairs),
                "gold_strictly_better_rate": round(wins / len(pairs), 4) if pairs else None,
                "tie_rate": round(ties / len(pairs), 4) if pairs else None,
                "auroc": (round(auroc, 4) if auroc is not None else None),
            }
        return out

    per_distr_sep = _summarize(per_distr)
    per_distr_sep_scale = _summarize(per_distr_scale)

    overall_easy_auroc = _auroc(gold_e, easy_e)
    overall_hard_auroc = _auroc(gold_e, hard_e)
    overall_easy_auroc_scale = _auroc(gold_e_s, easy_e_s)
    overall_hard_auroc_scale = _auroc(gold_e_s, hard_e_s)

    res = {
        "n_eval_test_inputs": n_eval,
        "n_eval_test_inputs_scale_subset": n_eval_scale,
        "n_scale_tasks": n_scale_tasks,
        "per_distractor_separation_ALL_tasks": per_distr_sep,
        "per_distractor_separation_SCALE_subset": per_distr_sep_scale,
        "overall_easy_auroc_ALL": (round(overall_easy_auroc, 4) if overall_easy_auroc is not None else None),
        "overall_hard_auroc_ALL": (round(overall_hard_auroc, 4) if overall_hard_auroc is not None else None),
        "overall_easy_auroc_SCALE": (round(overall_easy_auroc_scale, 4) if overall_easy_auroc_scale is not None else None),
        "overall_hard_auroc_SCALE": (round(overall_hard_auroc_scale, 4) if overall_hard_auroc_scale is not None else None),
    }

    # catches_hard on the SCALE subset (where the family fires): hard distractor kinds with
    # gold_strictly_better_rate >= 0.70.
    hard_kinds = ("perturbed_gold", "color_swap_gold", "wrong_dim_gold", "transposed_gold")
    catches_hard = [
        k for k in hard_kinds
        if per_distr_sep_scale.get(k, {}).get("gold_strictly_better_rate") is not None
        and per_distr_sep_scale[k]["gold_strictly_better_rate"] >= 0.70
    ]

    verdict = (
        f"complete: arc_tiling_scaling_invariant_scale_subset_n{n_eval_scale}_"
        f"perturbed{per_distr_sep_scale.get('perturbed_gold', {}).get('gold_strictly_better_rate')}_"
        f"transposed{per_distr_sep_scale.get('transposed_gold', {}).get('gold_strictly_better_rate')}_"
        f"colorswap{per_distr_sep_scale.get('color_swap_gold', {}).get('gold_strictly_better_rate')}_"
        f"catches{len(catches_hard)}hard"
    )

    art = {
        "experiment": "arc_invariant_tiling_scaling_draft",
        "title": "arc_tiling_scaling_structural_invariant",
        "family": "tiling_scaling",
        "honest_verdict": verdict,
        "inference_substrate": "rule_based_grid_verifier_against_cached_arc_solutions",
        "domain": "arc_agi1_grid",
        "split": split,
        "n_tasks": len(task_ids),
        "random_seed": seed,
        "label_source": "gold_arc_solutions",
        "results": res,
        "catches_hard": catches_hard,
        "no_llm_used": True,
        "no_induction": True,
        "no_gpu_used": True,
        "no_label_leak": "scorer uses only train pairs + candidate; test gold is never read into the score",
        "interpretation": (
            "The tiling/scaling invariant fires ONLY on scale-rule tasks (output dims = k*input "
            "dims). It verifies the candidate is a valid TILING/UPSCALING of the TEST INPUT under "
            "a train-selected structural model (subtile-transform with a FIXED 8-symmetry "
            "vocabulary, or block-upscale), scoring the fraction of subtiles/blocks that violate "
            "that decomposition. Unlike v1 (dims/palette/bg only), this is CONTENT-SENSITIVE: a "
            "perturbed gold breaks subtile equality. On non-scale tasks it abstains (0.0). "
            "The headline numbers are on the SCALE subset, where the family actually fires."
        ),
        "principle_label_source": (
            "gold ARC solutions are the oracle; distractors are cheap deterministic perturbations; "
            "the scorer is derived from train pairs only -> no LLM, no induction-by-search, no leak"
        ),
    }
    if write:
        OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


if __name__ == "__main__":
    a = run()
    print(f"-> {a['honest_verdict']}")
    r = a["results"]
    print(f"   n_eval(all)={r['n_eval_test_inputs']} n_eval(scale)={r['n_eval_test_inputs_scale_subset']} "
          f"n_scale_tasks={r['n_scale_tasks']}")
    print(f"   overall AUROC scale: easy={r['overall_easy_auroc_SCALE']} hard={r['overall_hard_auroc_SCALE']}")
    print("   --- per-distractor on SCALE subset (the family's domain) ---")
    for kind, d in r["per_distractor_separation_SCALE_subset"].items():
        print(f"     {kind:18s} n={d['n']:4d} gold_better={d['gold_strictly_better_rate']} "
              f"tie={d['tie_rate']} auroc={d['auroc']}")
    print("   --- per-distractor on ALL tasks (abstains on non-scale -> mostly ties) ---")
    for kind, d in r["per_distractor_separation_ALL_tasks"].items():
        print(f"     {kind:18s} n={d['n']:4d} gold_better={d['gold_strictly_better_rate']} "
              f"tie={d['tie_rate']} auroc={d['auroc']}")
    print(f"   catches_hard (scale subset, >=0.70): {a['catches_hard']}")
