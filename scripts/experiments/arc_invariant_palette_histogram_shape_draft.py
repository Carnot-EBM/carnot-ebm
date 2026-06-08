"""DRAFT (#C, north-star domain): a CHEAP, NON-INDUCING ARC verifier invariant family that
targets CONTENT errors v1 was blind to -- the palette_histogram_shape family.

WHY (north-star.md sec.0). ARC-AGI-3 is the north star; the verifier's job there is to make
search ACCURATE + EFFICIENT -- prune wrong candidate outputs cheaply so the generator/LLM is
called less. v1 (arc_grid_verifier_discriminator_draft.py) is a STRONG STRUCTURAL pruner
(dims/palette-membership/background) but BLIND to content errors that preserve structure:
its per-distractor gold-strictly-better rate was perturbed_gold 0.006, transposed 0.293,
color_swap 0.384, blank 0.135, copy_input 0.389. Those are the ceiling this family attacks.

WHAT IS NEW vs v1. v1's palette check is MEMBERSHIP only: "did a novel color appear?" (a set
test). This family looks at the per-color COUNT HISTOGRAM and its SHAPE. ARC train outputs
often preserve color-COUNT relationships, not just which colors are legal:
  * a background color dominates a stable fraction of the grid,
  * the rank-ordered count profile (sorted color frequencies, normalized) is stable,
  * the number of distinct colors is stable,
  * each color occupies a roughly-stable share of the non-background pixels.
We build a per-color and a rank-shape SIGNATURE from the TRAIN OUTPUTS ONLY (no test gold, no
rule induction, no search) and score a candidate by how far its histogram DEVIATES from that
signature. LOWER score = more consistent with the train-output histogram shape.

WHAT THIS TARGETS (the content errors):
  * blank (all background)         -> degenerate 1-color histogram: distinct-color count wrong,
                                      rank shape collapses, non-bg share = 0 -> large deviation.
  * random (uniform-ish 0-9)       -> too many distinct colors, flat rank profile, illegal
                                      colors carry mass -> large deviation.
  * color_swap_gold (counts move   -> per-color fractional composition no longer matches the
     between two colors)             train per-color signature, even though MEMBERSHIP is fine
                                      and the SORTED shape is identical. The per-color-identity
                                      term is the only one that can see this.

HARD CONSTRAINTS (honored): no LLM, no GPU, no induction-by-search, no test-gold label leak.
Everything is derived from the TRAIN pairs + the candidate grid. Deterministic.

  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    .venv/bin/python scripts/experiments/arc_invariant_palette_histogram_shape_draft.py
"""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/experiments")
from arc_grid_verifier_discriminator_draft import (  # noqa: E402  (path-insert before import)
    _auroc,
    _bg,
    _build_invariants,
    _colors,
    _dims,
    _distractors,
    _palette_counts,
    _predicted_dims,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARC = Path("/home/ianblenke/trm_src/kaggle/combined")
OUT = REPO_ROOT / "results" / "arc_invariant_palette_histogram_shape.json"

FAMILY = "palette_histogram_shape"


# ---------- histogram-signature helpers (all derived from TRAIN OUTPUTS only) ----------
def _norm_color_hist(g):
    """Per-color fractional composition: {color: count/total} over a grid's pixels."""
    cnt = _palette_counts(g)
    total = sum(cnt.values())
    if total == 0:
        return {}
    return {c: n / total for c, n in cnt.items()}


def _rank_profile(g, k=4):
    """Rank-ordered, normalized count SHAPE: sorted color fractions, descending, padded/cut to
    length k. This is palette-identity-INVARIANT -- it captures only the *shape* of the
    histogram (how concentrated vs flat the color mass is), which blank/random distort."""
    cnt = _palette_counts(g)
    total = sum(cnt.values())
    if total == 0:
        return [0.0] * k
    fr = sorted((n / total for n in cnt.values()), reverse=True)
    fr = (fr + [0.0] * k)[:k]
    return fr


def _build_hist_signature(train):
    """Build the palette_histogram_shape SIGNATURE from train OUTPUT grids only.

    Returns a dict with:
      mean_per_color : {color: mean fractional composition across train outputs}
                       (per-color IDENTITY signature -- the only term that sees color_swap)
      mean_rank      : mean rank-ordered normalized count profile (the SHAPE -- catches
                       blank's collapse and random's flatness, palette-identity-invariant)
      mean_distinct  : mean number of distinct colors per train output
      mean_bg_share  : mean fraction occupied by the (per-output) dominant color
      mean_nonbg_share_per_color : {color: mean share of NON-background pixels} -- a finer
                       per-color signature robust to background-area changes
    """
    k_rank = 6
    per_color_sum = Counter()
    rank_sum = [0.0] * k_rank
    distinct_sum = 0.0
    bg_share_sum = 0.0
    nonbg_per_color_sum = Counter()
    nonbg_pairs = 0  # number of train outputs that had any non-bg pixels (for averaging)
    n = 0
    for p in train:
        out = p["output"]
        cnt = _palette_counts(out)
        total = sum(cnt.values())
        if total == 0:
            continue
        n += 1
        for c, v in cnt.items():
            per_color_sum[c] += v / total
        rp = _rank_profile(out, k=k_rank)
        for i in range(k_rank):
            rank_sum[i] += rp[i]
        distinct_sum += len(cnt)
        bg = _bg(out)
        bg_share_sum += cnt[bg] / total
        nonbg_total = total - cnt[bg]
        if nonbg_total > 0:
            nonbg_pairs += 1
            for c, v in cnt.items():
                if c != bg:
                    nonbg_per_color_sum[c] += v / nonbg_total
    if n == 0:
        return None
    return {
        "k_rank": k_rank,
        "mean_per_color": {c: s / n for c, s in per_color_sum.items()},
        "mean_rank": [s / n for s in rank_sum],
        "mean_distinct": distinct_sum / n,
        "mean_bg_share": bg_share_sum / n,
        "mean_nonbg_share_per_color": (
            {c: s / nonbg_pairs for c, s in nonbg_per_color_sum.items()} if nonbg_pairs else {}
        ),
        "n_train": n,
    }


# ---------- the scorer: histogram-shape deviation energy (LOWER = more consistent) ----------
def _l1_over_union(sig_map, cand_map):
    """L1 distance between two {color: fraction} dicts over the UNION of their keys.
    Ranges [0, 2] (two disjoint unit-mass distributions). This is the per-color IDENTITY
    deviation -- a color_swap moves mass from color a to color b, so both the a-term and the
    b-term contribute, while a true gold matches the signature exactly."""
    keys = set(sig_map) | set(cand_map)
    if not keys:
        return 0.0
    return sum(abs(sig_map.get(c, 0.0) - cand_map.get(c, 0.0)) for c in keys)


def family_score(candidate_grid, inv, train_pairs, test_input):
    """palette_histogram_shape consistency-VIOLATION energy for ONE candidate grid.
    LOWER = more consistent with the train-output color-count histogram shape.

    Components (each in [0,1] after normalization; mean = final energy in [0,1]):
      per_color_dev : L1 over union of {color->fraction} vs train mean_per_color, /2 to [0,1].
                      SEES color_swap (membership-identical, composition-different).
      nonbg_dev     : same but over NON-background fractional shares -- finer per-color signal
                      robust to background-area shifts.
      rank_dev      : L1 between candidate rank-profile and train mean_rank, /2 to [0,1].
                      SEES blank (collapsed shape) + random (flat shape). Palette-invariant.
      distinct_dev  : |distinct_colors(cand) - mean_distinct| / max(mean_distinct, 1), clipped.
                      SEES blank (1 color) + random (~10 colors).
      bg_share_dev  : |bg_share(cand) - mean_bg_share|, in [0,1]. SEES blank (bg_share=1) +
                      random (bg_share ~ 0.1).

    Derived from TRAIN PAIRS + candidate ONLY -- no test gold, no induction, no search.
    """
    sig = _build_hist_signature(train_pairs)
    if sig is None:
        return 1.0  # no usable signature -> maximally-unsure -> high energy (consistent default)

    cnt = _palette_counts(candidate_grid)
    total = sum(cnt.values())
    if total == 0:
        return 1.0  # empty candidate -> maximal violation

    # per-color identity deviation (whole grid)
    cand_per_color = {c: v / total for c, v in cnt.items()}
    per_color_dev = _l1_over_union(sig["mean_per_color"], cand_per_color) / 2.0

    # per-color identity deviation over non-background pixels (finer; background-area robust)
    bg = _bg(candidate_grid)
    nonbg_total = total - cnt[bg]
    if nonbg_total > 0 and sig["mean_nonbg_share_per_color"]:
        cand_nonbg = {c: v / nonbg_total for c, v in cnt.items() if c != bg}
        nonbg_dev = _l1_over_union(sig["mean_nonbg_share_per_color"], cand_nonbg) / 2.0
    else:
        # candidate is single-color (blank-like) OR no train non-bg signature -> if train HAD
        # non-bg structure, a single-color candidate is maximally inconsistent on this axis.
        nonbg_dev = 1.0 if sig["mean_nonbg_share_per_color"] else 0.0

    # rank-shape deviation (palette-identity-invariant)
    cand_rank = _rank_profile(candidate_grid, k=sig["k_rank"])
    rank_dev = sum(abs(a - b) for a, b in zip(sig["mean_rank"], cand_rank)) / 2.0
    rank_dev = min(rank_dev, 1.0)

    # distinct-color-count deviation
    distinct_dev = min(abs(len(cnt) - sig["mean_distinct"]) / max(sig["mean_distinct"], 1.0), 1.0)

    # background-share deviation
    bg_share_dev = min(abs(cnt[bg] / total - sig["mean_bg_share"]), 1.0)

    feats = [per_color_dev, nonbg_dev, rank_dev, distinct_dev, bg_share_dev]
    return sum(feats) / len(feats)


# ---------- measurement on the SAME protocol as v1 ----------
def run(split="training", limit=None, seed=0, write=True):
    rng = random.Random(seed)
    ch = json.load(open(ARC / f"arc-agi_{split}_challenges.json"))
    so = json.load(open(ARC / f"arc-agi_{split}_solutions.json"))
    task_ids = list(ch)
    if limit:
        task_ids = task_ids[:limit]
    all_golds = [so[t][0] for t in task_ids if so.get(t)]

    n_eval = 0
    gold_scores, easy_scores, hard_scores = [], [], []
    # distractor_kind -> list of (gold_score, distractor_score)
    per_distr = {}
    easy_kinds = {"copy_input", "wrong_task_gold", "random", "blank"}
    hard_kinds = {"perturbed_gold", "color_swap_gold", "wrong_dim_gold", "transposed_gold"}

    for t in task_ids:
        task = ch[t]
        inv = _build_invariants(task["train"])
        train_pairs = task["train"]
        for ti, test in enumerate(task["test"]):
            if not so.get(t) or ti >= len(so[t]):
                continue
            gold = so[t][ti]
            tin = test["input"]
            easy, hard = _distractors(gold, tin, all_golds, rng)
            if not easy and not hard:
                continue
            n_eval += 1
            g_score = family_score(gold, inv, train_pairs, tin)
            gold_scores.append(-g_score)  # negate: higher = better for AUROC
            for kind, d in list(easy.items()) + list(hard.items()):
                d_score = family_score(d, inv, train_pairs, tin)
                per_distr.setdefault(kind, []).append((g_score, d_score))
                if kind in easy_kinds:
                    easy_scores.append(-d_score)
                elif kind in hard_kinds:
                    hard_scores.append(-d_score)

    # per-distractor: gold_strictly_better_rate (gold STRICTLY lower energy than distractor)
    # and a pairwise AUROC using -score as the ranking signal (gold should rank higher).
    per_distractor = {}
    for kind, pairs in sorted(per_distr.items()):
        wins = sum(1 for ge, de in pairs if ge < de)
        ties = sum(1 for ge, de in pairs if ge == de)
        pos = [-ge for ge, _ in pairs]  # gold (higher=better)
        neg = [-de for _, de in pairs]  # distractor
        au = _auroc(pos, neg)
        per_distractor[kind] = {
            "n": len(pairs),
            "gold_strictly_better_rate": round(wins / len(pairs), 4),
            "tie_rate": round(ties / len(pairs), 4),
            "auroc": (round(au, 4) if au is not None else None),
        }

    easy_auroc = _auroc(gold_scores, easy_scores)
    hard_auroc = _auroc(gold_scores, hard_scores)

    res = {
        "n_eval_test_inputs": n_eval,
        "easy_discrimination_auroc": (round(easy_auroc, 4) if easy_auroc is not None else None),
        "hard_discrimination_auroc": (round(hard_auroc, 4) if hard_auroc is not None else None),
        "per_distractor": per_distractor,
    }

    # catches_hard: hard distractor kinds where gold_strictly_better_rate >= 0.70
    catches_hard = sorted(
        k for k, v in per_distractor.items()
        if k in hard_kinds and v["gold_strictly_better_rate"] >= 0.70
    )
    res["catches_hard"] = catches_hard

    # honest comparison to v1 hard-distractor ceiling (from results/arc_grid_verifier_discriminator.json)
    v1_hard = {
        "perturbed_gold": 0.0056,
        "color_swap_gold": 0.384,
        "wrong_dim_gold": 0.863,
        "transposed_gold": 0.2927,
    }
    beats_v1 = {
        k: {
            "this": per_distractor.get(k, {}).get("gold_strictly_better_rate"),
            "v1": v1_hard[k],
            "delta": (round(per_distractor[k]["gold_strictly_better_rate"] - v1_hard[k], 4)
                      if k in per_distractor else None),
        }
        for k in v1_hard
    }
    res["vs_v1_hard"] = beats_v1

    art = {
        "experiment": f"arc_invariant_{FAMILY}_draft",
        "title": f"arc_invariant_{FAMILY}_content_error_verifier",
        "family": FAMILY,
        "inference_substrate": "rule_based_grid_verifier_against_cached_arc_solutions",
        "domain": "arc_agi1_grid",
        "split": split,
        "n_tasks": len(task_ids),
        "random_seed": seed,
        "label_source": "gold_arc_solutions",
        "no_llm_used": True,
        "no_induction": True,
        "no_test_gold_in_score": True,
        "results": res,
        "honest_verdict": _verdict(res, catches_hard),
        "interpretation": (
            "palette_histogram_shape goes beyond v1's palette-MEMBERSHIP set test: it compares "
            "the candidate's per-color count histogram AND its rank-ordered shape against a "
            "signature built from TRAIN OUTPUTS only. It is designed to catch content errors "
            "that preserve structure: blank (collapsed histogram), random (flat/illegal "
            "histogram), and color_swap (per-color composition mismatch). Whether it beats v1 "
            "on the HARD content distractors is reported honestly in vs_v1_hard / catches_hard."
        ),
    }
    if write:
        OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


def _verdict(res, catches_hard):
    ea = res["easy_discrimination_auroc"]
    ha = res["hard_discrimination_auroc"]
    if catches_hard:
        tag = "catches_" + "_".join(catches_hard)
    else:
        tag = "no_hard_distractor_above_0.70_ceiling_holds"
    return (f"complete: arc_invariant_{FAMILY}_{tag}"
            f"_easyAUROC{ea}_hardAUROC{ha}_n{res['n_eval_test_inputs']}")


if __name__ == "__main__":
    a = run()
    r = a["results"]
    print(f"-> {a['honest_verdict']}")
    print(f"   family={a['family']} n_eval={r['n_eval_test_inputs']}")
    print(f"   EASY auroc={r['easy_discrimination_auroc']}  HARD auroc={r['hard_discrimination_auroc']}")
    print(f"   catches_hard={r['catches_hard']}")
    print("   per_distractor (kind: gold_strictly_better_rate / auroc):")
    for kind, v in sorted(r["per_distractor"].items()):
        print(f"     {kind:18s} rate={v['gold_strictly_better_rate']:.4f} "
              f"auroc={v['auroc']} tie={v['tie_rate']:.4f} n={v['n']}")
    print("   vs_v1_hard (this - v1 on hard distractors):")
    for kind, v in sorted(r["vs_v1_hard"].items()):
        print(f"     {kind:18s} this={v['this']} v1={v['v1']} delta={v['delta']}")
