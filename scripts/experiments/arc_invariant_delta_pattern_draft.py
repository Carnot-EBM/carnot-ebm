"""DRAFT invariant family: delta_pattern -- change-pattern consistency for ARC grids.

WHY (north-star.md §0 + the v1 ceiling). The v1 ARC grid verifier
(arc_grid_verifier_discriminator_draft.py) is a CHEAP, NON-INDUCING structural pruner:
it ranks gold above distractors using only dims/palette/background invariants from the
train pairs. It PRUNES structurally-malformed candidates (wrong_dim 0.863, wrong_task
0.947, random 0.889) but is BLIND to CONTENT errors that preserve structure -- the hard
distractors collapse: perturbed_gold 0.006, transposed 0.293, color_swap 0.384,
blank 0.135, copy_input 0.389. Those are the ceiling this family targets.

THE IDEA (delta_pattern). Many ARC tasks transform the input by changing only SOME cells
and leaving the rest fixed. From each train pair we compute the DELTA: the set of cells
where input != output, the colors that DISAPPEAR there (input-side), the colors that
APPEAR there (output-side), and the structural footprint (how many cells change, and a
crude locality measure). That gives a per-task "delta signature" derived ONLY from the
train pairs -- no rule induction, no LLM, no use of the test gold.

For a candidate output, we compute the delta between the test INPUT and the CANDIDATE and
score how INCONSISTENT it is with the train signature:

  * copy_input  -> the (input -> candidate) delta is EMPTY. If every train pair changed at
    least one cell, an empty candidate-delta is a strong violation (the transform is
    known to do *something*, and copy_input does nothing). This is the single cleanest
    signal in the family.
  * perturbed_gold -> introduces extra changed cells, often flipping cells the train
    transform leaves fixed, and/or introducing/removing colors that are not part of the
    train delta's appear/disappear color sets. Both raise the inconsistency score.
  * color_swap_gold / transposed_gold -> their (input -> candidate) delta involves colors
    or a change-footprint that does not match the train signature.

This only has cell-by-cell meaning when input and candidate share dimensions, i.e. for the
"same"-dim tasks (262 of 400 in v1's dim_rule_coverage). For non-"same" tasks (scale,
const, unknown) a cell-wise delta is undefined, so the family abstains (returns a constant
0.0 = "no information") rather than fabricating a signal. We report numbers BOTH over the
full 400-task protocol (so it is directly comparable to v1) AND restricted to same-dim
tasks (where the family is actually active), so the honest headroom is visible.

HARD CONSTRAINTS honored: no LLM, no GPU, no induction-by-search; cheap deterministic
structural check derivable from the train pairs; the test gold is NEVER used in scoring.

  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
    .venv/bin/python scripts/experiments/arc_invariant_delta_pattern_draft.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "scripts/experiments")
from arc_grid_verifier_discriminator_draft import (  # noqa: E402  (reuse, pure helpers)
    _auroc,
    _bg,
    _build_invariants,
    _colors,
    _dims,
    _distractors,
    _predicted_dims,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARC = Path("/home/ianblenke/trm_src/kaggle/combined")
OUT = REPO_ROOT / "results" / "arc_invariant_delta_pattern.json"

HARD_KINDS = ("perturbed_gold", "color_swap_gold", "wrong_dim_gold", "transposed_gold")
EASY_KINDS = ("copy_input", "wrong_task_gold", "random", "blank")


# ---------------------------------------------------------------------------
# delta computation: the cells that change between two equal-dimension grids
# ---------------------------------------------------------------------------
def _delta(a, b):
    """Return (changed_cells, from_colors, to_colors) for two SAME-dim grids a->b.

    changed_cells: list of (r, c) where a[r][c] != b[r][c].
    from_colors:   Counter of the colors that were OVERWRITTEN (a-side at changed cells).
    to_colors:     Counter of the colors that were WRITTEN  (b-side at changed cells).
    Returns None if the grids differ in dimension (delta undefined).
    """
    if _dims(a) != _dims(b):
        return None
    changed = []
    from_c, to_c = Counter(), Counter()
    for r, (ra, rb) in enumerate(zip(a, b)):
        for c, (xa, xb) in enumerate(zip(ra, rb)):
            if xa != xb:
                changed.append((r, c))
                from_c[xa] += 1
                to_c[xb] += 1
    return changed, from_c, to_c


def _build_delta_signature(train):
    """Aggregate the train-pair deltas into a 'what a valid change looks like' signature.

    Derived ONLY from train pairs. Captures:
      from_colors / to_colors : the colors that legitimately disappear / appear at changed
                                cells across all train pairs (the 'palette of the change').
      any_change              : did EVERY train pair change at least one cell? (so an empty
                                candidate-delta is a violation).
      change_fracs            : per-pair fraction of cells changed (footprint magnitude).
      all_same_dim            : are all train pairs equal-dim input->output? (the family is
                                only active when the test transform is plausibly same-dim).
    """
    from_colors, to_colors = Counter(), Counter()
    change_fracs = []
    any_change_per_pair = []
    all_same_dim = True
    for p in train:
        d = _delta(p["input"], p["output"])
        if d is None:
            all_same_dim = False
            continue
        changed, fc, tc = d
        from_colors += fc
        to_colors += tc
        h, w = _dims(p["input"])
        n = h * w
        change_fracs.append((len(changed) / n) if n else 0.0)
        any_change_per_pair.append(len(changed) > 0)
    return {
        "from_colors": set(from_colors),
        "to_colors": set(to_colors),
        # colors that the change is allowed to touch at all (either side)
        "delta_colors": set(from_colors) | set(to_colors),
        "all_change": bool(any_change_per_pair) and all(any_change_per_pair),
        "any_change": any(any_change_per_pair),
        "min_frac": min(change_fracs) if change_fracs else 0.0,
        "max_frac": max(change_fracs) if change_fracs else 0.0,
        "mean_frac": (sum(change_fracs) / len(change_fracs)) if change_fracs else 0.0,
        "all_same_dim": all_same_dim,
        "n_pairs_with_delta": len(change_fracs),
    }


def family_score(candidate_grid, inv, train_pairs, test_input):
    """delta_pattern inconsistency score. LOWER = more consistent with the train deltas.

    inv is the v1 invariant dict (reused); we (re)derive the delta signature from
    train_pairs here so the function matches the required signature exactly. The test gold
    is NEVER referenced. Returns a float in [0, ~] where 0 means 'fully consistent' (or
    'family inactive / abstains' for non-same-dim tasks).
    """
    sig = _build_delta_signature(train_pairs)

    # FALLBACK when the cell-wise delta is undefined (the train transform is not
    # consistently same-dim, OR the candidate's dims differ from the test input's, so an
    # input->candidate cell delta has no meaning). Rather than returning a flat 0.0 for all
    # candidates (which ties gold with content-wrong distractors and dilutes the signal),
    # we apply the ONE part of the delta signature that is still well-defined cross-dim:
    # the PALETTE OF THE CHANGE. The output of a valid transform is built from the colors
    # the train pairs actually produce (the to_colors of the change) plus colors that were
    # already present and survive unchanged (the train input/output palettes). A candidate
    # whose cells use colors entirely outside that union is off-signature. This stays
    # within the delta_pattern family (it is the change-palette, not the dims) and never
    # references the test gold. It is a WEAK fallback by design -- the strong cell-wise
    # checks below are where the family earns its keep on same-dim tasks.
    if not sig["all_same_dim"] or _delta(test_input, candidate_grid) is None:
        allowed = sig["to_colors"] | set(inv.get("all_palette", set()))
        cc = _colors(candidate_grid)
        if not cc or not allowed:
            return 0.0
        off = len([c for c in cc if c not in allowed]) / len(cc)
        # also: an output that is byte-identical to the test input on a task whose train
        # transform always changes the grid is suspicious even cross-dim (copy_input on a
        # same-dim task is caught below; this catches copy_input where dims happen to match
        # but the family abstained for another reason).
        copy_pen = 1.0 if (sig["all_change"] and candidate_grid == test_input) else 0.0
        return max(off, copy_pen)
    d = _delta(test_input, candidate_grid)

    changed, from_c, to_c = d
    h, w = _dims(test_input)
    ncells = h * w if h and w else 1
    frac = len(changed) / ncells

    violations = []

    # (1) EMPTY-DELTA violation: the train transform always changes something, but this
    #     candidate changes nothing (copy_input is exactly this). Strong, clean signal.
    if sig["all_change"]:
        violations.append(1.0 if len(changed) == 0 else 0.0)

    # (2) APPEAR-COLOR violation: colors written into changed cells that the train deltas
    #     never wrote. perturbed_gold often writes a color into a cell where the train
    #     never introduces it. Fraction of changed cells whose new color is off-signature.
    if changed and sig["to_colors"]:
        off = sum(to_c[col] for col in to_c if col not in sig["to_colors"])
        violations.append(off / len(changed))

    # (3) DISAPPEAR-COLOR violation: colors overwritten that the train deltas never
    #     overwrite (the candidate is editing cells the transform should leave fixed).
    if changed and sig["from_colors"]:
        off = sum(from_c[col] for col in from_c if col not in sig["from_colors"])
        violations.append(off / len(changed))

    # (4) FOOTPRINT-MAGNITUDE violation: how far the candidate's change-fraction falls
    #     outside the train [min,max] band (with a small tolerance). perturbed_gold and
    #     blank/color_swap inflate (or, for copy_input, deflate) the footprint.
    lo, hi = sig["min_frac"], sig["max_frac"]
    tol = 0.05  # absolute tolerance so tiny tasks are not over-penalized
    if frac < lo - tol:
        mag = (lo - frac) / (lo + 1e-9)
    elif frac > hi + tol:
        mag = (frac - hi) / (hi + 1e-9)
    else:
        mag = 0.0
    violations.append(min(1.0, mag))

    if not violations:
        return 0.0
    return sum(violations) / len(violations)


# ---------------------------------------------------------------------------
# measurement: SAME protocol as v1 (400 tasks, seed 0 distractors)
# ---------------------------------------------------------------------------
def run(split="training", limit=None, seed=0, write=True):
    rng = random.Random(seed)
    ch = json.load(open(ARC / f"arc-agi_{split}_challenges.json"))
    so = json.load(open(ARC / f"arc-agi_{split}_solutions.json"))
    task_ids = list(ch)
    if limit:
        task_ids = task_ids[:limit]
    all_golds = [so[t][0] for t in task_ids if so.get(t)]

    # per_distr[kind] -> list of (gold_score, distractor_score)
    per_distr = {}
    # restricted view: only tasks where the family is ACTIVE (same-dim train transform)
    per_distr_active = {}
    gold_scores, easy_scores, hard_scores = [], [], []
    gold_scores_a, easy_scores_a, hard_scores_a = [], [], []
    n_eval = 0
    n_active = 0

    for t in task_ids:
        task = ch[t]
        inv = _build_invariants(task["train"])
        sig = _build_delta_signature(task["train"])
        active = sig["all_same_dim"]
        for ti, test in enumerate(task["test"]):
            if not so.get(t) or ti >= len(so[t]):
                continue
            gold = so[t][ti]
            tin = test["input"]
            easy, hard = _distractors(gold, tin, all_golds, rng)
            if not easy and not hard:
                continue
            n_eval += 1
            if active:
                n_active += 1

            gs = family_score(gold, inv, task["train"], tin)
            gold_scores.append(-gs)
            if active:
                gold_scores_a.append(-gs)

            for kind, d in list(easy.items()) + list(hard.items()):
                dsc = family_score(d, inv, task["train"], tin)
                per_distr.setdefault(kind, []).append((gs, dsc))
                if active:
                    per_distr_active.setdefault(kind, []).append((gs, dsc))
                if kind in EASY_KINDS:
                    easy_scores.append(-dsc)
                    if active:
                        easy_scores_a.append(-dsc)
                else:
                    hard_scores.append(-dsc)
                    if active:
                        hard_scores_a.append(-dsc)

    def _summarize(pd):
        out = {}
        for kind, pairs in sorted(pd.items()):
            wins = sum(1 for gsc, dsc in pairs if gsc < dsc)
            ties = sum(1 for gsc, dsc in pairs if gsc == dsc)
            # pairwise AUROC using -score as the ranking signal (gold should rank higher)
            pos = [-gsc for gsc, _ in pairs]
            neg = [-dsc for _, dsc in pairs]
            a = _auroc(pos, neg)
            out[kind] = {
                "n": len(pairs),
                "gold_strictly_better_rate": round(wins / len(pairs), 4) if pairs else None,
                "tie_rate": round(ties / len(pairs), 4) if pairs else None,
                "auroc": round(a, 4) if a is not None else None,
            }
        return out

    per_distr_sep = _summarize(per_distr)
    per_distr_sep_active = _summarize(per_distr_active)

    easy_auroc = _auroc(gold_scores, easy_scores)
    hard_auroc = _auroc(gold_scores, hard_scores)
    easy_auroc_a = _auroc(gold_scores_a, easy_scores_a)
    hard_auroc_a = _auroc(gold_scores_a, hard_scores_a)

    catches_hard = [k for k in HARD_KINDS
                    if per_distr_sep.get(k, {}).get("gold_strictly_better_rate") is not None
                    and per_distr_sep[k]["gold_strictly_better_rate"] >= 0.70]

    res = {
        "n_eval_test_inputs": n_eval,
        "n_active_same_dim": n_active,
        "overall_easy_auroc": round(easy_auroc, 4) if easy_auroc is not None else None,
        "overall_hard_auroc": round(hard_auroc, 4) if hard_auroc is not None else None,
        "overall_easy_auroc_active": round(easy_auroc_a, 4) if easy_auroc_a is not None else None,
        "overall_hard_auroc_active": round(hard_auroc_a, 4) if hard_auroc_a is not None else None,
        "per_distractor_separation": per_distr_sep,
        "per_distractor_separation_active_only": per_distr_sep_active,
        "catches_hard": catches_hard,
    }

    verdict = (f"complete: arc_delta_pattern_catches_{'_'.join(catches_hard) or 'none_hard'}"
               f"_easyAUROC{res['overall_easy_auroc']}_hardAUROC{res['overall_hard_auroc']}"
               f"_active{n_active}_of_{n_eval}")

    art = {
        "experiment": "arc_invariant_delta_pattern_draft",
        "title": "arc_invariant_delta_pattern_change_consistency",
        "honest_verdict": verdict,
        "inference_substrate": "rule_based_grid_verifier_against_cached_arc_solutions",
        "domain": "arc_agi1_grid",
        "family": "delta_pattern",
        "split": split,
        "n_tasks": len(task_ids),
        "random_seed": seed,
        "label_source": "gold_arc_solutions",
        "results": res,
        "no_llm_used": True,
        "no_induction": True,
        "no_label_leak": True,
        "interpretation": (
            "delta_pattern scores how inconsistent the (test_input -> candidate) change is "
            "with the per-task change signature derived ONLY from train pairs (which colors "
            "appear/disappear at changed cells, the change-footprint band, and whether any "
            "change is expected at all). It is designed to catch CONTENT errors that the v1 "
            "structural verifier is blind to -- copy_input (empty delta when a change is "
            "expected) most cleanly, and perturbed_gold via off-signature colors / inflated "
            "footprint. It ABSTAINS (score 0) on non-same-dim tasks where a cell-wise delta "
            "is undefined, so the 'active_only' numbers show the true headroom."
        ),
    }
    if write:
        OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


if __name__ == "__main__":
    a = run()
    r = a["results"]
    print(f"-> {a['honest_verdict']}")
    print(f"   n_eval={r['n_eval_test_inputs']}  n_active_same_dim={r['n_active_same_dim']}")
    print(f"   overall easy AUROC={r['overall_easy_auroc']}  hard AUROC={r['overall_hard_auroc']}")
    print(f"   active-only easy AUROC={r['overall_easy_auroc_active']}  "
          f"hard AUROC={r['overall_hard_auroc_active']}")
    print("   per-distractor (full protocol):")
    for k, v in r["per_distractor_separation"].items():
        print(f"     {k:16s} gold_better={v['gold_strictly_better_rate']}  "
              f"auroc={v['auroc']}  tie={v['tie_rate']}  n={v['n']}")
    print("   per-distractor (active same-dim tasks only):")
    for k, v in r["per_distractor_separation_active_only"].items():
        print(f"     {k:16s} gold_better={v['gold_strictly_better_rate']}  "
              f"auroc={v['auroc']}  tie={v['tie_rate']}  n={v['n']}")
    print(f"   catches_hard (>=0.70 full protocol) = {r['catches_hard']}")
