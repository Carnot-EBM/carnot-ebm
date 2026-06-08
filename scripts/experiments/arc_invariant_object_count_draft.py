"""DRAFT (#C, north-star domain): the OBJECT_COUNT invariant family — a CHEAP, NON-INDUCING
ARC verifier designed to catch CONTENT errors that the v1 dims/palette/bg verifier is BLIND to.

WHY (north-star.md §0). ARC-AGI-3 is the north star; the verifier's job is to PRUNE wrong
candidate outputs cheaply so the generator/LLM is called less (the efficiency axis). The v1
verifier (arc_grid_verifier_discriminator_draft.py) ranks gold above STRUCTURALLY-malformed
distractors (wrong dims/palette/bg) well — wrong_task_gold 0.947, random 0.889, wrong_dim
0.863 — but is BLIND to CONTENT errors that preserve structure: perturbed_gold 0.006,
transposed 0.293, color_swap 0.384, blank 0.135, copy_input 0.389. Those are the ceiling.

THE OBJECT_COUNT FAMILY. Most ARC outputs are not random pixel soup: they are a small number
of connected colored "objects" (4-neighbour connected components, per non-background color)
with a characteristic count, size distribution, and color multiset. The TRAIN OUTPUTS reveal
this pattern (e.g. "every output has exactly 3 objects of size 4", or "outputs have 5-9
objects, sizes in {1,2,4}, colors {2,3}"). We build a SIGNATURE from the train-output object
statistics ONLY (no test gold, no rule induction, no search) and score a candidate by how far
its own object statistics deviate from that signature. LOWER score = more consistent.

WHAT IT TARGETS (vs v1's blind spots):
  * blank: an all-background candidate has ZERO non-bg objects -> max count deviation.
  * perturbed_gold: flipping ~12% of cells FRAGMENTS objects (extra singletons) and shifts
    the size distribution -> count + size-distribution deviation.
  * copy_input: the input's object structure usually differs from the output's signature.
  * color_swap_gold: changes the per-color object multiset (objects move between colors).
  * transposed_gold: count/sizes are TRANSPOSE-INVARIANT, so this family is EXPECTED to be
    weak here (honest limitation — transpose preserves component structure exactly).

This is a CHEAP deterministic structural check derivable from the train pairs alone. No LLM,
no GPU, no rule-induction-by-search, no test-gold leak.

  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    .venv/bin/python scripts/experiments/arc_invariant_object_count_draft.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

# Reuse the v1 pure helpers (no heavy deps). _build_invariants gives us the inferred bg;
# _distractors gives the SAME candidate set + kinds as v1 so the comparison is apples-to-apples.
from arc_grid_verifier_discriminator_draft import (  # noqa: E402
    _auroc,
    _bg,
    _build_invariants,
    _colors,
    _distractors,
    _dims,
)

ARC = Path("/home/ianblenke/trm_src/kaggle/combined")
OUT = REPO_ROOT / "results" / "arc_invariant_object_count.json"


# ---------- connected-component object statistics (4-neighbour, per non-bg color) ----------
def _objects(grid, bg):
    """Return list of connected components (4-neighbour) of equal non-background color.

    Each component is summarised as (color, size). We do NOT keep pixel coordinates — the
    family is about COUNT / SIZE / COLOR of objects, which is what an ARC output's structure
    is usually made of. Background (bg) cells are skipped entirely; a "blank" grid that is
    all background therefore yields ZERO objects (the load-bearing signal for the blank
    distractor). Cheap: single linear scan + iterative flood fill, O(H*W).
    """
    h, w = _dims(grid)
    if h == 0 or w == 0:
        return []
    seen = [[False] * w for _ in range(h)]
    comps = []
    for r0 in range(h):
        for c0 in range(w):
            if seen[r0][c0]:
                continue
            color = grid[r0][c0]
            if color == bg:
                seen[r0][c0] = True
                continue
            # iterative flood fill over same-color 4-neighbours
            stack = [(r0, c0)]
            seen[r0][c0] = True
            size = 0
            while stack:
                r, c = stack.pop()
                size += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            comps.append((color, size))
    return comps


def _object_stats(grid, bg):
    """Compact object-statistics feature vector for a grid given a background color."""
    comps = _objects(grid, bg)
    sizes = [s for _, s in comps]
    colors = [c for c, _ in comps]
    return {
        "count": len(comps),                                  # number of objects
        "sizes": tuple(sorted(sizes)),                        # multiset of object sizes
        "size_set": frozenset(sizes),                         # distinct object sizes
        "color_set": frozenset(colors),                       # distinct object colors
        "color_count_multiset": Counter(colors),              # objects per color
        "max_size": max(sizes) if sizes else 0,
        "mean_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
        "total_fg": sum(sizes),                               # total non-bg cells
    }


# ---------- build the train-OUTPUT object-stats signature (no induction) ----------
def _build_object_signature(train, inv):
    """Summarise the object statistics of the TRAIN OUTPUT grids into a signature.

    The signature is purely descriptive of what train outputs look like — it does NOT model
    HOW the output is produced from the input (that would be induction). We record the
    observed range of object counts, the union of object sizes / colors seen, and the
    distribution of objects-per-color. A candidate is later scored by deviation from this.

    Background choice: use the v1-inferred bg when train outputs agree on one; else fall back
    to per-grid most-common color. This keeps the family aligned with v1's notion of bg.
    """
    fixed_bg = inv.get("bg")
    counts, all_sizes, all_colors = [], set(), set()
    per_color_counts = []  # list of Counter(color -> n_objects) per train output
    size_multisets = []    # list of tuple(sorted sizes) per train output
    for p in train:
        out = p["output"]
        bg = fixed_bg if fixed_bg is not None else _bg(out)
        st = _object_stats(out, bg)
        counts.append(st["count"])
        all_sizes |= st["size_set"]
        all_colors |= st["color_set"]
        per_color_counts.append(st["color_count_multiset"])
        size_multisets.append(st["sizes"])

    # objects-per-color: is it constant across train outputs? (common ARC pattern)
    color_count_consistent = len(set(tuple(sorted(c.items())) for c in per_color_counts)) == 1
    # size multiset: is it constant across train outputs?
    size_multiset_consistent = len(set(size_multisets)) == 1

    return {
        "bg": fixed_bg,
        "count_min": min(counts) if counts else 0,
        "count_max": max(counts) if counts else 0,
        "count_set": frozenset(counts),
        "count_constant": (len(set(counts)) == 1),
        "size_union": frozenset(all_sizes),          # sizes EVER seen in train outputs
        "color_union": frozenset(all_colors),        # object colors EVER seen
        "size_multiset_consistent": size_multiset_consistent,
        "canonical_size_multiset": size_multisets[0] if size_multiset_consistent and size_multisets else None,
        "color_count_consistent": color_count_consistent,
        "canonical_color_counts": per_color_counts[0] if color_count_consistent and per_color_counts else None,
        "max_train_object_size": max((s for ms in size_multisets for s in ms), default=0),
    }


# ---------- the scorer: object-stats deviation energy (LOWER = more consistent) ----------
def family_score(candidate_grid, inv, train_pairs, test_input):
    """object_count family score: distance between candidate object-stats and the train-OUTPUT
    object-stats signature. LOWER = more consistent with how train outputs are structured.

    Components (each in [0,1], averaged):
      count_dev   — how far the candidate's object count is outside the train-output count
                    range, normalised. ZERO objects (blank) lands far outside [min,max] when
                    train outputs have objects -> the load-bearing blank signal.
      size_dev    — fraction of candidate object sizes never seen in any train output. A
                    perturbed grid sprouts singleton fragments of novel sizes -> raised.
      color_dev   — fraction of candidate object colors never used as an object color in any
                    train output. color_swap moves mass into a color that may be absent.
      multiset_dev— if train outputs share a CONSTANT size multiset, penalise candidates whose
                    size multiset differs (exact structural match expected). Skipped (0.0)
                    when train outputs disagree, so we never over-claim a constraint that
                    isn't there.
      colorcnt_dev— if train outputs share a CONSTANT objects-per-color signature, penalise
                    candidates that deviate (L1 over the color->count vector, normalised).
    """
    sig = _build_object_signature(train_pairs, inv)
    bg = sig["bg"] if sig["bg"] is not None else _bg(candidate_grid)
    st = _object_stats(candidate_grid, bg)

    # --- count deviation: distance outside the observed train count range, normalised ---
    cmin, cmax = sig["count_min"], sig["count_max"]
    cnt = st["count"]
    if cmin <= cnt <= cmax:
        count_dev = 0.0
    else:
        dist = (cmin - cnt) if cnt < cmin else (cnt - cmax)
        # normalise by a scale that keeps blank (cnt=0, cmin>=1) clearly positive but bounded
        scale = max(cmax, 1)
        count_dev = min(1.0, dist / scale)

    # --- size deviation: candidate object sizes never seen in train outputs ---
    cand_sizes = st["size_set"]
    if cand_sizes:
        novel_sizes = sum(1 for s in cand_sizes if s not in sig["size_union"])
        size_dev = novel_sizes / len(cand_sizes)
    else:
        # candidate has NO objects: maximally novel size-profile only if train outputs HAD
        # objects (else the empty-output is genuinely consistent -> 0).
        size_dev = 1.0 if sig["size_union"] else 0.0

    # --- color deviation: candidate object colors never used as object colors in train ---
    cand_colors = st["color_set"]
    if cand_colors:
        novel_colors = sum(1 for c in cand_colors if c not in sig["color_union"])
        color_dev = novel_colors / len(cand_colors)
    else:
        color_dev = 1.0 if sig["color_union"] else 0.0

    # --- exact size-multiset match (only when train outputs agree on one) ---
    if sig["size_multiset_consistent"] and sig["canonical_size_multiset"] is not None:
        multiset_dev = 0.0 if st["sizes"] == sig["canonical_size_multiset"] else 1.0
    else:
        multiset_dev = 0.0  # no constant constraint to enforce -> don't penalise

    # --- exact objects-per-color match (only when train outputs agree on one) ---
    if sig["color_count_consistent"] and sig["canonical_color_counts"] is not None:
        want = sig["canonical_color_counts"]
        got = st["color_count_multiset"]
        keys = set(want) | set(got)
        l1 = sum(abs(want.get(k, 0) - got.get(k, 0)) for k in keys)
        denom = sum(want.values()) + sum(got.values())
        colorcnt_dev = (l1 / denom) if denom else 0.0
    else:
        colorcnt_dev = 0.0

    feats = [count_dev, size_dev, color_dev, multiset_dev, colorcnt_dev]
    return sum(feats) / len(feats)


# ---------- evaluation harness: SAME protocol as v1 ----------
def run(split="training", limit=None, seed=0, write=True):
    rng = random.Random(seed)
    ch = json.load(open(ARC / f"arc-agi_{split}_challenges.json"))
    so = json.load(open(ARC / f"arc-agi_{split}_solutions.json"))
    task_ids = list(ch)
    if limit:
        task_ids = task_ids[:limit]
    all_golds = [so[t][0] for t in task_ids if so.get(t)]

    n_eval = 0
    gold_e_easy, gold_e_hard = [], []  # gold scores paired (for overall AUROC)
    easy_scores, hard_scores = [], []
    per_distr = {}  # kind -> list of (gold_score, distractor_score)

    for t in task_ids:
        task = ch[t]
        inv = _build_invariants(task["train"])
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
            for kind, d in list(easy.items()) + list(hard.items()):
                ds = family_score(d, inv, task["train"], tin)
                per_distr.setdefault(kind, []).append((gs, ds))
            for kind, d in easy.items():
                easy_scores.append(family_score(d, inv, task["train"], tin))
                gold_e_easy.append(gs)
            for kind, d in hard.items():
                hard_scores.append(family_score(d, inv, task["train"], tin))
                gold_e_hard.append(gs)

    # per-distractor: gold_strictly_better_rate (gold score STRICTLY lower) + pairwise AUROC.
    # Ranking signal = -score (so a lower score => higher rank). AUROC over (gold pos vs this
    # distractor neg) using -score; gold should rank ABOVE the distractor.
    per_distr_out = {}
    EASY_KINDS = {"copy_input", "wrong_task_gold", "random", "blank"}
    for kind, pairs in sorted(per_distr.items()):
        wins = sum(1 for gs, ds in pairs if gs < ds)
        ties = sum(1 for gs, ds in pairs if gs == ds)
        pos = [-gs for gs, _ in pairs]   # gold (higher -score = better)
        neg = [-ds for _, ds in pairs]   # distractor
        auroc = _auroc(pos, neg)
        per_distr_out[kind] = {
            "n": len(pairs),
            "gold_strictly_better_rate": round(wins / len(pairs), 4),
            "tie_rate": round(ties / len(pairs), 4),
            "auroc": (round(auroc, 4) if auroc is not None else None),
        }

    # overall easy/hard AUROC: gold (pos) vs distractor (neg), ranking by -score
    easy_auroc = _auroc([-x for x in gold_e_easy], [-x for x in easy_scores])
    hard_auroc = _auroc([-x for x in gold_e_hard], [-x for x in hard_scores])

    # catches_hard: hard distractor kinds where gold_strictly_better_rate >= 0.70
    HARD_KINDS = {"perturbed_gold", "color_swap_gold", "wrong_dim_gold", "transposed_gold"}
    catches_hard = sorted(
        k for k in HARD_KINDS
        if k in per_distr_out and per_distr_out[k]["gold_strictly_better_rate"] >= 0.70
    )

    # v1 ceiling for honest comparison (from results/arc_grid_verifier_discriminator.json)
    v1_hard = {
        "perturbed_gold": 0.0056,
        "color_swap_gold": 0.384,
        "transposed_gold": 0.2927,
        "wrong_dim_gold": 0.863,
        "blank": 0.1349,
        "copy_input": 0.3894,
    }
    beats_v1 = {
        k: (per_distr_out[k]["gold_strictly_better_rate"] - v1_hard[k])
        for k in v1_hard if k in per_distr_out
    }

    res = {
        "family": "object_count",
        "n_eval_test_inputs": n_eval,
        "per_distractor": per_distr_out,
        "overall_easy_auroc": (round(easy_auroc, 4) if easy_auroc is not None else None),
        "overall_hard_auroc": (round(hard_auroc, 4) if hard_auroc is not None else None),
        "catches_hard": catches_hard,
        "delta_vs_v1_gold_strictly_better_rate": {k: round(v, 4) for k, v in beats_v1.items()},
    }

    # honest verdict
    if catches_hard:
        token = "catches_" + "_".join(catches_hard)
    else:
        token = "no_hard_catch_above_0p70"
    verdict = (
        f"complete: arc_invariant_object_count_{token}"
        f"_easyAUROC{res['overall_easy_auroc']}_hardAUROC{res['overall_hard_auroc']}_n{n_eval}"
    )

    art = {
        "experiment": "arc_invariant_object_count_draft",
        "title": "arc_object_count_invariant_family_discriminator",
        "honest_verdict": verdict,
        "inference_substrate": "rule_based_grid_verifier_against_cached_arc_solutions",
        "domain": "arc_agi1_grid",
        "family": "object_count",
        "split": split,
        "n_tasks": len(task_ids),
        "random_seed": seed,
        "label_source": "gold_arc_solutions",
        "no_llm_used": True,
        "no_induction": True,
        "no_test_gold_leak": True,
        "results": res,
        "interpretation": (
            "object_count derives a per-task object-statistics signature (connected-component "
            "count / size-multiset / per-color object counts) from the TRAIN OUTPUTS only, then "
            "scores a candidate by deviation from that signature. It is designed to catch CONTENT "
            "errors v1 is blind to: blank (zero objects) and perturbed_gold (fragmentation shifts "
            "the count/size profile). It is EXPECTED to stay weak on transposed_gold (component "
            "count/sizes are transpose-invariant). Honest comparison vs the v1 ceiling is in "
            "delta_vs_v1_gold_strictly_better_rate."
        ),
    }
    if write:
        OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


if __name__ == "__main__":
    a = run()
    print(f"-> {a['honest_verdict']}")
    r = a["results"]
    print(f"   n={r['n_eval_test_inputs']}")
    print(f"   overall easy_auroc={r['overall_easy_auroc']} hard_auroc={r['overall_hard_auroc']}")
    print(f"   catches_hard={r['catches_hard']}")
    print("   per_distractor (gold_strictly_better_rate, auroc):")
    for k in sorted(r["per_distractor"]):
        d = r["per_distractor"][k]
        print(f"     {k:18s} rate={d['gold_strictly_better_rate']:.4f} auroc={d['auroc']} n={d['n']}")
    print("   delta_vs_v1 (object_count - v1):")
    for k, v in sorted(r["delta_vs_v1_gold_strictly_better_rate"].items()):
        print(f"     {k:18s} {v:+.4f}")
