"""Combined ARC cheap-verifier (v2): union/ensemble of the 7 invariant families + v1.

WHY (north-star.md §0). ARC-AGI-3 is the north star; the verifier PRUNES wrong candidate
outputs cheaply so the generator/LLM is invoked less (the efficiency axis). The v1 verifier
(arc_grid_verifier_discriminator_draft.py: dims/palette/bg) is a STRUCTURAL pruner that is
BLIND to CONTENT errors that preserve structure (perturbed_gold 0.006, color_swap 0.384,
transposed 0.293). Seven new invariant families were measured individually; each is a
LOW-COVERAGE specialist that breaks the v1 content-blindness only on the slice of ARC where
its precondition fires. This harness measures whether COMBINING them (each gated to fire only
where applicable) lifts the per-distractor gold_strictly_better_rate above the v1 ceiling on
the full 400-task protocol.

HONEST COMBINER DESIGN (the load-bearing methodology choice). The families have DIFFERENT
abstain conventions:
  - symmetry / color_mapping / content_overlap abstain at the NEUTRAL value 0.5,
  - tiling_scaling abstains at 0.0 (its BEST score — a naive min() would let an abstaining
    tiling family always claim 'perfectly consistent' and dominate),
  - delta_pattern abstains with a weak palette fallback (often near 0.0),
  - object_count / palette_histogram_shape never truly abstain (always compute a deviation).
A naive min/max over RAW family scores is therefore WRONG. Instead we compute, per task, an
APPLICABILITY flag for each family from the TRAIN PAIRS ONLY (the same precondition the family
uses internally, independent of the candidate), and AGGREGATE ONLY over families that are
applicable on that task. v1 is always applicable (the structural floor).

Combiners reported (all reuse the exact per-family family_score; no re-implementation):
  1. union_max     : violation = max over {v1} ∪ {applicable content families}.
                     Rationale: gold should have LOW violation on EVERY applicable check; a
                     distractor is caught if ANY applicable check flags it. This is the honest
                     'ensemble of specialists' pruner. Primary headline combiner.
  2. mean_defined  : violation = mean over {v1} ∪ {applicable content families}. Softer; less
                     prone to a single noisy family hurting gold, but dilutes a strong catch.
  3. logistic_cv   : a small logistic regression over the 8 family scores (v1 + 7), trained
                     with 5-fold cross-validation so the reported numbers are OUT-OF-FOLD (no
                     optimistic in-sample bias). This is the LEARNED-ENSEMBLE UPPER BOUND on
                     what these cheap features can extract; it is NOT a deployable rule (it is
                     fit to the distractor distribution), so it is reported as an estimate of
                     the achievable ceiling, clearly labelled.

No LLM, no GPU, no rule-induction-by-search, no test-gold leak (every family_score already
verified leak-free; this harness only calls them + combines their outputs). Deterministic:
random.Random(0), identical distractor protocol to v1.

  JAX_PLATFORMS=cpu .venv/bin/python \
    scripts/experiments/arc_grid_verifier_invariants_v2_combined.py
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

# v1 structural verifier: shared harness helpers + the structural violation energy.
from arc_grid_verifier_discriminator_draft import (  # noqa: E402
    _auroc,
    _bg,
    _build_invariants,
    _colors,
    _dims,
    _distractors,
    _energy,
    _violation_features,
)

# Each family's scorer + (where needed) its internal applicability helper.
import arc_invariant_symmetry_draft as F_sym  # noqa: E402
import arc_invariant_color_mapping_draft as F_cm  # noqa: E402
import arc_invariant_object_count_draft as F_oc  # noqa: E402
import arc_invariant_content_overlap_draft as F_co  # noqa: E402
import arc_invariant_delta_pattern_draft as F_dp  # noqa: E402
import arc_invariant_tiling_scaling_draft as F_ts  # noqa: E402
import arc_invariant_palette_histogram_shape_draft as F_ph  # noqa: E402

ARC = Path("/home/ianblenke/trm_src/kaggle/combined")
OUT = REPO_ROOT / "results" / "arc_grid_verifier_invariants_v2.json"

V1_HARD = {
    "perturbed_gold": 0.0056,
    "transposed_gold": 0.2927,
    "color_swap_gold": 0.384,
    "wrong_dim_gold": 0.863,
    "copy_input": 0.3894,
    "blank": 0.1349,
    "random": 0.8889,
    "wrong_task_gold": 0.947,
}
HARD_KINDS = ("perturbed_gold", "color_swap_gold", "transposed_gold", "wrong_dim_gold")
EASY_KINDS = ("copy_input", "wrong_task_gold", "random", "blank")


# --------------------------------------------------------------------------------------
# Per-family applicability, computed from the TRAIN PAIRS only (candidate-independent).
# This mirrors each family's OWN internal abstain precondition so that "applicable" means
# exactly "this family did NOT fall back to its neutral score". Families that never abstain
# (object_count, palette_histogram_shape) are always applicable.
# --------------------------------------------------------------------------------------
def _applicability(inv, train_pairs, test_input):
    app = {}
    # symmetry: a single dihedral transform consistent across ALL train pairs exists.
    app["symmetry"] = F_sym._consistent_transform(train_pairs) is not None
    # color_mapping: a consistent cellwise colour map M was inferred (>=90% train agreement).
    app["color_mapping"] = F_cm._infer_color_map(train_pairs) is not None
    # content_overlap: at least one cell-level signal (mode-template OR copy-expectation) is defined.
    app["content_overlap"] = (
        F_co._mode_template(train_pairs) is not None
        or F_co._copy_expectation(train_pairs, test_input) is not None
    )
    # delta_pattern: the train transform is consistently same-dim (cell-wise delta defined).
    app["delta_pattern"] = bool(F_dp._build_delta_signature(train_pairs)["all_same_dim"])
    # tiling_scaling: the dimension rule is an integer scale.
    app["tiling_scaling"] = inv["dim_rule"][0] == "scale"
    # always-on content families (never abstain to a neutral tie):
    app["object_count"] = True
    app["palette_histogram_shape"] = True
    return app


# Order matters only for the logistic feature vector; keep it stable.
FAMILY_ORDER = (
    "symmetry",
    "color_mapping",
    "object_count",
    "content_overlap",
    "delta_pattern",
    "tiling_scaling",
    "palette_histogram_shape",
)
FAMILY_SCORERS = {
    "symmetry": F_sym.family_score,
    "color_mapping": F_cm.family_score,
    "object_count": F_oc.family_score,
    "content_overlap": F_co.family_score,
    "delta_pattern": F_dp.family_score,
    "tiling_scaling": F_ts.family_score,
    "palette_histogram_shape": F_ph.family_score,
}


def _v1_violation(cand, inv, test_input):
    return _energy(_violation_features(cand, inv, test_input))


def _combined_scores(cand, inv, train_pairs, test_input, app):
    """Return (union_max, mean_defined, feature_vector) for one candidate.

    union_max / mean_defined aggregate v1 (always) + every APPLICABLE content family.
    feature_vector is [v1, then each family in FAMILY_ORDER] using the family's RAW score
    when applicable and its neutral value (0.5) when not — for the logistic, which learns
    its own weights including a learned 'ignore when neutral' behaviour.
    """
    v1 = _v1_violation(cand, inv, test_input)
    defined = [v1]
    feats = [v1]
    for name in FAMILY_ORDER:
        raw = FAMILY_SCORERS[name](cand, inv, train_pairs, test_input)
        if app[name]:
            defined.append(raw)
            feats.append(raw)
        else:
            feats.append(0.5)  # neutral placeholder for the logistic feature row
    union_max = max(defined)
    mean_defined = sum(defined) / len(defined)
    return union_max, mean_defined, feats


# --------------------------------------------------------------------------------------
# Minimal logistic regression (no sklearn dependency), L2-regularised, 5-fold CV.
# Label: 1 = distractor (should be PRUNED / high violation), 0 = gold (should be kept).
# We report OUT-OF-FOLD predictions so the per-distractor numbers are not in-sample-optimistic.
# --------------------------------------------------------------------------------------
def _standardize(X):
    n, d = len(X), len(X[0])
    mean = [0.0] * d
    for row in X:
        for j in range(d):
            mean[j] += row[j]
    mean = [m / n for m in mean]
    var = [0.0] * d
    for row in X:
        for j in range(d):
            var[j] += (row[j] - mean[j]) ** 2
    std = [math.sqrt(v / n) or 1.0 for v in var]
    Z = [[(row[j] - mean[j]) / std[j] for j in range(d)] for row in X]
    return Z, mean, std


def _fit_logreg(X, y, l2=1.0, lr=0.2, iters=400):
    n, d = len(X), len(X[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(iters):
        gw = [0.0] * d
        gb = 0.0
        for i in range(n):
            z = b + sum(w[j] * X[i][j] for j in range(d))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            err = p - y[i]
            gb += err
            for j in range(d):
                gw[j] += err * X[i][j]
        b -= lr * (gb / n)
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j] / n)
    return w, b


def _predict_proba(w, b, x):
    z = b + sum(w[j] * x[j] for j in range(len(x)))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _logistic_cv_oof(rows, k=5, seed=0):
    """rows: list of (feature_vec, label, group_idx, kind, is_gold). Returns oof prob per row.

    Folds are split by GROUP (test-input index) so a task's gold + all its distractors land in
    the SAME fold — prevents the model from memorising a task across folds.
    """
    groups = sorted({r[2] for r in rows})
    rng = random.Random(seed)
    rng.shuffle(groups)
    fold_of = {g: (i % k) for i, g in enumerate(groups)}
    oof = [None] * len(rows)
    for f in range(k):
        tr = [(r[0], r[1]) for r in rows if fold_of[r[2]] != f]
        teidx = [i for i, r in enumerate(rows) if fold_of[r[2]] == f]
        if not tr or not teidx:
            continue
        Xtr = [a for a, _ in tr]
        ytr = [bb for _, bb in tr]
        Z, mean, std = _standardize(Xtr)
        w, b = _fit_logreg(Z, ytr)
        for i in teidx:
            xz = [(rows[i][0][j] - mean[j]) / std[j] for j in range(len(mean))]
            oof[i] = _predict_proba(w, b, xz)
    return oof


# --------------------------------------------------------------------------------------
# Main evaluation: identical distractor protocol to v1 (seed 0, same iteration order).
# --------------------------------------------------------------------------------------
def run(split="training", limit=None, seed=0, write=True):
    t0 = time.time()
    rng = random.Random(seed)
    ch = json.load(open(ARC / f"arc-agi_{split}_challenges.json"))
    so = json.load(open(ARC / f"arc-agi_{split}_solutions.json"))
    task_ids = list(ch)
    if limit:
        task_ids = task_ids[:limit]
    all_golds = [so[t][0] for t in task_ids if so.get(t)]

    # per (combiner, distractor_kind) -> list[(gold_score, distractor_score)]
    pd_union, pd_mean = {}, {}
    # logistic needs a flat row table; we also keep gold/distractor pairing by group.
    log_rows = []        # (feature_vec, label, group_idx, kind, is_gold)
    group_kinds = {}     # group_idx -> {kind: distractor_row_index}
    group_gold = {}      # group_idx -> gold_row_index
    # overall easy/hard for union combiner
    gold_e_easy_u, gold_e_hard_u, easy_u, hard_u = [], [], [], []
    app_counts = {name: 0 for name in FAMILY_ORDER}
    n_eval = 0

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
            gi = n_eval
            n_eval += 1
            app = _applicability(inv, task["train"], tin)
            for name in FAMILY_ORDER:
                if app[name]:
                    app_counts[name] += 1

            gu, gm, gfeat = _combined_scores(gold, inv, task["train"], tin, app)
            log_rows.append((gfeat, 0, gi, "GOLD", True))
            group_gold[gi] = len(log_rows) - 1
            group_kinds.setdefault(gi, {})

            for kind, d in list(easy.items()) + list(hard.items()):
                du, dm, dfeat = _combined_scores(d, inv, task["train"], tin, app)
                pd_union.setdefault(kind, []).append((gu, du))
                pd_mean.setdefault(kind, []).append((gm, dm))
                log_rows.append((dfeat, 1, gi, kind, False))
                group_kinds[gi][kind] = len(log_rows) - 1
                if kind in EASY_KINDS:
                    gold_e_easy_u.append(gu)
                    easy_u.append(du)
                else:
                    gold_e_hard_u.append(gu)
                    hard_u.append(du)

    def _per_distr_table(pd_pairs):
        out = {}
        for kind, pairs in sorted(pd_pairs.items()):
            wins = sum(1 for gs, ds in pairs if gs < ds)
            ties = sum(1 for gs, ds in pairs if gs == ds)
            pos = [-gs for gs, _ in pairs]
            neg = [-ds for _, ds in pairs]
            a = _auroc(pos, neg)
            out[kind] = {
                "n": len(pairs),
                "gold_strictly_better_rate": round(wins / len(pairs), 4),
                "tie_rate": round(ties / len(pairs), 4),
                "auroc": (round(a, 4) if a is not None else None),
            }
        return out

    union_tbl = _per_distr_table(pd_union)
    mean_tbl = _per_distr_table(pd_mean)

    # ---- logistic CV (out-of-fold) ----
    oof = _logistic_cv_oof(log_rows, k=5, seed=seed)
    # build per-distractor pairs using oof prob as the (signed) score: higher prob = more
    # 'distractor-like' = higher violation, so for gold_strictly_better we want gold prob LOWER.
    pd_log = {}
    for gi, gold_ri in group_gold.items():
        gp = oof[gold_ri]
        if gp is None:
            continue
        for kind, dri in group_kinds[gi].items():
            dp = oof[dri]
            if dp is None:
                continue
            pd_log.setdefault(kind, []).append((gp, dp))
    log_tbl = _per_distr_table(pd_log)

    # overall easy/hard AUROC for the union combiner
    eu = _auroc([-x for x in gold_e_easy_u], [-x for x in easy_u])
    hu = _auroc([-x for x in gold_e_hard_u], [-x for x in hard_u])

    def _catches(tbl):
        return sorted(
            k for k in HARD_KINDS
            if k in tbl and tbl[k]["gold_strictly_better_rate"] >= 0.70
        )

    def _delta_vs_v1(tbl):
        return {
            k: round(tbl[k]["gold_strictly_better_rate"] - V1_HARD[k], 4)
            for k in V1_HARD if k in tbl
        }

    results = {
        "n_eval_test_inputs": n_eval,
        "family_applicability_counts": {
            **{k: app_counts[k] for k in FAMILY_ORDER},
            "v1_structural": n_eval,
            "_pct_of_test_inputs": {
                k: round(app_counts[k] / n_eval, 3) for k in FAMILY_ORDER
            },
        },
        "v1_baseline_per_distractor": V1_HARD,
        "combiner_union_max": {
            "description": "max violation over v1 + applicable content families (ensemble pruner)",
            "per_distractor": union_tbl,
            "overall_easy_auroc": (round(eu, 4) if eu is not None else None),
            "overall_hard_auroc": (round(hu, 4) if hu is not None else None),
            "catches_hard_ge_0p70": _catches(union_tbl),
            "delta_vs_v1_gold_strictly_better_rate": _delta_vs_v1(union_tbl),
        },
        "combiner_mean_defined": {
            "description": "mean violation over v1 + applicable content families",
            "per_distractor": mean_tbl,
            "catches_hard_ge_0p70": _catches(mean_tbl),
            "delta_vs_v1_gold_strictly_better_rate": _delta_vs_v1(mean_tbl),
        },
        "combiner_logistic_cv_oof": {
            "description": (
                "5-fold group-CV logistic over [v1, 7 family scores]; OUT-OF-FOLD predictions. "
                "This is a LEARNED-ENSEMBLE UPPER BOUND on the cheap features (fit to the "
                "distractor distribution), an ESTIMATE of the achievable ceiling, not a "
                "deployable rule."
            ),
            "per_distractor": log_tbl,
            "catches_hard_ge_0p70": _catches(log_tbl),
            "delta_vs_v1_gold_strictly_better_rate": _delta_vs_v1(log_tbl),
        },
    }

    wall = round(time.time() - t0, 2)
    catches = _catches(union_tbl)
    log_catches = _catches(log_tbl)
    token = ("union_catches_" + "_".join(catches)) if catches else "union_no_hard_catch_ge_0p70"
    verdict = (
        f"complete: arc_combined_invariant_verifier_v2_{token}"
        f"_logistic_catches_{('_'.join(log_catches) if log_catches else 'none')}"
        f"_n{n_eval}_wall{wall}s"
    )

    art = {
        "experiment": "arc_grid_verifier_invariants_v2_combined",
        "title": "arc_combined_cheap_invariant_verifier_content_ceiling",
        "honest_verdict": verdict,
        "inference_substrate": "rule_based_grid_verifier_against_cached_arc_solutions",
        "domain": "arc_agi1_grid",
        "split": split,
        "n_tasks": len(task_ids),
        "random_seed": seed,
        "label_source": "gold_arc_solutions",
        "no_llm_used": True,
        "no_induction": True,
        "no_test_gold_leak": True,
        "submitted_to_leaderboard": False,
        "wall_time_s": wall,
        "results": results,
    }
    if write:
        OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


if __name__ == "__main__":
    a = run()
    print(f"-> {a['honest_verdict']}")
    r = a["results"]
    print(f"   n={r['n_eval_test_inputs']} wall={a['wall_time_s']}s")
    print("   applicability (% of test inputs):")
    for k, v in r["family_applicability_counts"]["_pct_of_test_inputs"].items():
        print(f"     {k:24s} {v:.3f}")
    for cname in ("combiner_union_max", "combiner_mean_defined", "combiner_logistic_cv_oof"):
        c = r[cname]
        print(f"\n   == {cname} == catches_hard={c['catches_hard_ge_0p70']}")
        for k in ("perturbed_gold", "color_swap_gold", "transposed_gold", "wrong_dim_gold",
                  "copy_input", "blank", "random", "wrong_task_gold"):
            if k in c["per_distractor"]:
                d = c["per_distractor"][k]
                v1 = a["results"]["v1_baseline_per_distractor"].get(k)
                print(f"     {k:18s} rate={d['gold_strictly_better_rate']:.4f} "
                      f"auroc={d['auroc']} (v1 {v1})")
