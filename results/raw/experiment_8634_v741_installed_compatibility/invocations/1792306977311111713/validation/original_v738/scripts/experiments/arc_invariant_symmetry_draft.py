"""DRAFT (#C, north-star domain): SYMMETRY invariant family for the ARC grid verifier.

WHY (north-star.md §0). The v1 cheap verifier
(arc_grid_verifier_discriminator_draft.py) PRUNES structurally-malformed candidates
(wrong dim/palette/background) well but is BLIND to CONTENT errors that preserve
structure: perturbed_gold 0.006, transposed 0.293, color_swap 0.384 gold-strictly-better.
THOSE are the ceiling this family attacks.

THE IDEA (cheap, deterministic, NON-inducing). Some ARC tasks are pure GEOMETRIC
transforms: output = T(input) where T is one of a small fixed group of grid symmetries
(identity, horizontal flip, vertical flip, 180 rotation, transpose, anti-transpose,
90-CW, 90-CCW). We do NOT search/induce an arbitrary rule -- we only CHECK whether ONE
member of this fixed 8-element group is consistent with EVERY train (input,output) pair.
If a consistent transform T holds across ALL train pairs, then the gold test output MUST
equal T(test_input). The score is the normalized cell-disagreement between the candidate
and T(test_input).

This is exactly the kind of CONTENT check v1 lacks: when T applies, the score depends on
the ACTUAL CELL VALUES of the candidate, so:
  * perturbed_gold differs from T(test_input) in the perturbed cells -> caught.
  * color_swap differs wherever the two swapped colors appear -> caught.
  * transposed_gold (a wrong geometry) differs from the train-consistent geometry -> caught.
  * blank / copy_input / random differ massively -> caught.

HONEST LIMITATION (stated up front, measured below). Most ARC tasks are NOT pure
geometric transforms. For those, NO member of the fixed group is consistent across the
train pairs, and the family has NO content signal -- it must BACK OFF (emit a neutral
constant) rather than fabricate discrimination. So the family's lift is concentrated on
the SUBSET of tasks that ARE symmetry tasks; over all 400 tasks the average lift is
diluted by the back-off tasks. The measurement reports BOTH the all-tasks numbers
(apples-to-apples vs v1) AND the symmetry-applicable subset so the dilution is visible.

NO label leak: the score uses ONLY the train pairs + the candidate + the test_input. The
gold test output is never read inside family_score.

  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    .venv/bin/python scripts/experiments/arc_invariant_symmetry_draft.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARC = Path("/home/ianblenke/trm_src/kaggle/combined")
OUT = REPO_ROOT / "results" / "arc_invariant_symmetry.json"

# Reuse the v1 draft's pure helpers (no heavy deps).
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))
from arc_grid_verifier_discriminator_draft import (  # noqa: E402
    _build_invariants,
    _predicted_dims,
    _distractors,
    _dims,
    _colors,
    _bg,
    _auroc,
)


# ---------- the fixed symmetry group (8 dihedral transforms) ----------
# Each is a pure function grid -> grid. No parameters, no search beyond this fixed set.
def _identity(g):
    return [row[:] for row in g]


def _flip_h(g):  # mirror left-right (reverse each row)
    return [row[::-1] for row in g]


def _flip_v(g):  # mirror top-bottom (reverse row order)
    return [row[:] for row in g[::-1]]


def _rot180(g):
    return [row[::-1] for row in g[::-1]]


def _transpose(g):  # main diagonal
    h, w = _dims(g)
    return [[g[r][c] for r in range(h)] for c in range(w)]


def _anti_transpose(g):  # anti-diagonal = transpose then rot180
    return _rot180(_transpose(g))


def _rot90_cw(g):  # 90 degrees clockwise
    h, w = _dims(g)
    return [[g[h - 1 - r][c] for r in range(h)] for c in range(w)]


def _rot90_ccw(g):  # 90 degrees counter-clockwise
    h, w = _dims(g)
    return [[g[r][w - 1 - c] for r in range(h)] for c in range(w)]


# Ordered so that identity is first (preferred when several happen to tie on a pair).
_TRANSFORMS = [
    ("identity", _identity),
    ("flip_h", _flip_h),
    ("flip_v", _flip_v),
    ("rot180", _rot180),
    ("transpose", _transpose),
    ("anti_transpose", _anti_transpose),
    ("rot90_cw", _rot90_cw),
    ("rot90_ccw", _rot90_ccw),
]


def _consistent_transform(train):
    """Return the name of the symmetry transform that maps EVERY train input to its train
    output EXACTLY, or None if no single member of the fixed group is consistent.

    This is a CHECK over a fixed 8-element group, NOT a search over arbitrary rules: it is
    O(8 * |train| * cells) and cannot induce anything outside pure dihedral symmetry.
    """
    for name, fn in _TRANSFORMS:
        ok = True
        for p in train:
            if fn(p["input"]) != p["output"]:
                ok = False
                break
        if ok:
            return name
    return None


_TFN = dict(_TRANSFORMS)


def _cell_disagreement(a, b):
    """Normalized fraction of disagreeing cells between two grids.

    If dims differ, the grids are aligned on the overlap and every out-of-overlap cell on
    EITHER side counts as a disagreement (so a wrong-dim candidate is heavily penalized).
    Returns a value in [0, 1]; 0.0 means identical.
    """
    ah, aw = _dims(a)
    bh, bw = _dims(b)
    H, W = max(ah, bh), max(aw, bw)
    if H == 0 or W == 0:
        return 1.0
    disagree = 0
    for r in range(H):
        for c in range(W):
            av = a[r][c] if r < ah and c < aw else None
            bv = b[r][c] if r < bh and c < bw else None
            if av != bv:
                disagree += 1
    return disagree / (H * W)


# Neutral back-off score for tasks where no symmetry transform applies. Chosen at the
# MIDDLE of the [0,1] disagreement range so that, on non-symmetry tasks, gold and every
# distractor get the SAME score (a true tie -> contributes 0 to gold_strictly_better_rate
# and 0.5 to AUROC; i.e. the family honestly abstains rather than fabricating signal).
_BACKOFF = 0.5


def family_score(candidate_grid, inv, train_pairs, test_input):
    """SYMMETRY-family consistency score. LOWER = more consistent with the train pairs.

    1. Find the symmetry transform T (from the fixed 8-element dihedral group) that is
       consistent across ALL train pairs. If none, ABSTAIN (return the neutral back-off).
    2. Otherwise the train-consistent prediction is T(test_input). Score the candidate by
       its normalized cell-disagreement against that prediction.

    No gold test output is used. No rule induction beyond the fixed symmetry group.
    """
    tname = _consistent_transform(train_pairs)
    if tname is None:
        return _BACKOFF
    predicted = _TFN[tname](test_input)
    return _cell_disagreement(candidate_grid, predicted)


def run(split="training", limit=None, seed=0, write=True):
    rng = random.Random(seed)
    ch = json.load(open(ARC / f"arc-agi_{split}_challenges.json"))
    so = json.load(open(ARC / f"arc-agi_{split}_solutions.json"))
    task_ids = list(ch)
    if limit:
        task_ids = task_ids[:limit]
    all_golds = [so[t][0] for t in task_ids if so.get(t)]

    n_eval = 0
    gold_e, distr_e = [], []  # all-tasks scores (negated -> higher = better, for AUROC)
    # per_distr[kind] = list of (gold_score, distractor_score) over all evaluated test inputs
    per_distr = {}
    # subset where the symmetry transform actually applied
    per_distr_sym = {}
    n_symmetry_tasks = 0
    sym_transform_dist = Counter()

    for t in task_ids:
        task = ch[t]
        inv = _build_invariants(task["train"])
        tname = _consistent_transform(task["train"])
        applies = tname is not None
        if applies:
            n_symmetry_tasks += 1
            sym_transform_dist[tname] += 1
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
                distr_e.append(-ds)
                if applies:
                    per_distr_sym.setdefault(kind, []).append((gs, ds))

    def _summarize(pd):
        out = {}
        for kind, pairs in sorted(pd.items()):
            wins = sum(1 for gsc, dsc in pairs if gsc < dsc)
            ties = sum(1 for gsc, dsc in pairs if gsc == dsc)
            pos = [-gsc for gsc, _ in pairs]
            neg = [-dsc for _, dsc in pairs]
            auroc = _auroc(pos, neg)
            out[kind] = {
                "n": len(pairs),
                "gold_strictly_better_rate": round(wins / len(pairs), 4),
                "tie_rate": round(ties / len(pairs), 4),
                "auroc": round(auroc, 4) if auroc is not None else None,
            }
        return out

    per_distr_all = _summarize(per_distr)
    per_distr_sym_sum = _summarize(per_distr_sym)

    # split distractors into easy/hard for overall AUROC, matching v1 taxonomy
    EASY = {"copy_input", "wrong_task_gold", "random", "blank"}
    HARD = {"perturbed_gold", "color_swap_gold", "wrong_dim_gold", "transposed_gold"}
    easy_pos, easy_neg, hard_pos, hard_neg = [], [], [], []
    for kind, pairs in per_distr.items():
        for gsc, dsc in pairs:
            if kind in EASY:
                easy_pos.append(-gsc)
                easy_neg.append(-dsc)
            elif kind in HARD:
                hard_pos.append(-gsc)
                hard_neg.append(-dsc)
    overall_easy_auroc = _auroc(easy_pos, easy_neg)
    overall_hard_auroc = _auroc(hard_pos, hard_neg)

    catches_hard = sorted(
        k for k in HARD
        if k in per_distr_all and per_distr_all[k]["gold_strictly_better_rate"] >= 0.70
    )

    res = {
        "n_eval_test_inputs": n_eval,
        "n_tasks": len(task_ids),
        "n_symmetry_applicable_tasks": n_symmetry_tasks,
        "symmetry_transform_distribution": dict(sym_transform_dist),
        "overall_easy_auroc": round(overall_easy_auroc, 4) if overall_easy_auroc else None,
        "overall_hard_auroc": round(overall_hard_auroc, 4) if overall_hard_auroc else None,
        "per_distractor_all_tasks": per_distr_all,
        "per_distractor_symmetry_subset": per_distr_sym_sum,
        "catches_hard": catches_hard,
    }

    verdict = (
        f"complete: arc_invariant_symmetry_n{n_eval}"
        f"_symtasks{n_symmetry_tasks}of{len(task_ids)}"
        f"_easyAUROC{res['overall_easy_auroc']}_hardAUROC{res['overall_hard_auroc']}"
        f"_catches_hard{len(catches_hard)}"
    )
    art = {
        "experiment": "arc_invariant_symmetry_draft",
        "title": "arc_grid_symmetry_invariant_content_verifier",
        "family": "symmetry",
        "honest_verdict": verdict,
        "inference_substrate": "rule_based_grid_verifier_against_cached_arc_solutions",
        "domain": "arc_agi1_grid",
        "split": split,
        "n_tasks": len(task_ids),
        "random_seed": seed,
        "label_source": "gold_arc_solutions",
        "results": res,
        "no_llm_used": True,
        "no_induction_beyond_fixed_symmetry_group": True,
        "no_label_leak": True,
        "interpretation": (
            "The symmetry family checks whether ONE member of a fixed 8-element dihedral "
            "group (identity / flips / rotations / transposes) maps every train input to "
            "its train output. When it does, the gold test output must equal T(test_input), "
            "so any CONTENT error (perturbed, color-swapped) or wrong geometry (transposed) "
            "shows up as cell-disagreement -- the content signal v1 lacks. When no transform "
            "is consistent (most tasks), the family ABSTAINS with a neutral tie score rather "
            "than fabricating discrimination. Lift is therefore concentrated on the "
            "symmetry-applicable subset; over all 400 tasks it is diluted by abstention."
        ),
    }
    if write:
        OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


if __name__ == "__main__":
    a = run()
    print(f"-> {a['honest_verdict']}")
    r = a["results"]
    print(f"   n_eval={r['n_eval_test_inputs']}  symmetry_tasks={r['n_symmetry_applicable_tasks']}/{r['n_tasks']}")
    print(f"   transform_dist={r['symmetry_transform_distribution']}")
    print(f"   overall easy AUROC={r['overall_easy_auroc']}  hard AUROC={r['overall_hard_auroc']}")
    print("   --- per-distractor (ALL tasks: gold_strictly_better / auroc) ---")
    for k, v in r["per_distractor_all_tasks"].items():
        print(f"     {k:18s} gsb={v['gold_strictly_better_rate']:.4f} auroc={v['auroc']} (n={v['n']})")
    print("   --- per-distractor (SYMMETRY-applicable subset only) ---")
    for k, v in r["per_distractor_symmetry_subset"].items():
        print(f"     {k:18s} gsb={v['gold_strictly_better_rate']:.4f} auroc={v['auroc']} (n={v['n']})")
    print(f"   catches_hard (>=0.70 on ALL tasks) = {r['catches_hard']}")
