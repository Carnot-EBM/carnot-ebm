"""DRAFT (#C, north-star domain): can a CHEAP, NON-INDUCING verifier PRUNE wrong ARC
candidate outputs? — the verifier-as-router/pruner role measured on REAL ARC data.

WHY (north-star.md §0). ARC-AGI-3 is the north star; the verifier's job there is NOT to
induce the rule (the generator does that) but to make search ACCURATE + EFFICIENT —
prune obviously-wrong candidate outputs cheaply so the generator/LLM is called less
(the efficiency axis). This experiment instantiates that on the LOCAL public ARC-AGI-1
corpus (400 training tasks, gold solutions present at ~/trm_src/kaggle/combined): given
a test input and a candidate output grid, score CONSISTENCY with invariants derived ONLY
from the task's train pairs (dimension rule, palette, background) — no rule induction, no
LLM. Then ask: does the verifier rank the GOLD output above DISTRACTORS?

Honest expectation (the north-star division of labor, ARC instantiation):
  * EASY distractors (copy-input, wrong-task-gold, random, blank) -> cheap verifier
    PRUNES them reliably (the efficiency win: don't waste generator actions on
    dimension/palette-wrong candidates).
  * HARD distractors (perturbed gold, transposed, color-swapped, wrong-dim) -> cheap
    verifier hits a CEILING; catching these needs actual rule-induction (generator's
    job). This is the ARC analog of math being self-consistency-bound and code being
    execution-bound: VERIFIER PRUNES, GENERATOR INDUCES.

  .venv/bin/python scripts/experiments/arc_grid_verifier_discriminator_draft.py
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARC = Path("/home/ianblenke/trm_src/kaggle/combined")
OUT = REPO_ROOT / "results" / "arc_grid_verifier_discriminator.json"


# ---------- grid helpers (grids are list[list[int]], colors 0-9) ----------
def _dims(g):
    return (len(g), len(g[0]) if g else 0)


def _colors(g):
    return {c for row in g for c in row}


def _palette_counts(g):
    return Counter(c for row in g for c in row)


def _bg(g):
    c = _palette_counts(g)
    return c.most_common(1)[0][0] if c else 0


# ---------- invariant model derived ONLY from train pairs ----------
def _build_invariants(train):
    """Infer cheap, NON-inducing structural rules from (input,output) train pairs."""
    in_dims = [_dims(p["input"]) for p in train]
    out_dims = [_dims(p["output"]) for p in train]
    out_palette, in_palette, bgs = set(), set(), []
    for p in train:
        out_palette |= _colors(p["output"])
        in_palette |= _colors(p["input"])
        bgs.append(_bg(p["output"]))

    # dimension rule: same | integer-scale | constant | unknown
    dim_rule = ("unknown", None)
    if all(o == i for o, i in zip(out_dims, in_dims)):
        dim_rule = ("same", None)
    elif all(i[0] and i[1] and o[0] % i[0] == 0 and o[1] % i[1] == 0
             and (o[0] // i[0], o[1] // i[1]) == (out_dims[0][0] // in_dims[0][0],
                                                  out_dims[0][1] // in_dims[0][1])
             for o, i in zip(out_dims, in_dims)):
        dim_rule = ("scale", (out_dims[0][0] // in_dims[0][0], out_dims[0][1] // in_dims[0][1]))
    elif len(set(out_dims)) == 1:
        dim_rule = ("const", out_dims[0])

    return {
        "dim_rule": dim_rule,
        "out_palette": out_palette,
        "in_palette": in_palette,
        "all_palette": out_palette | in_palette,
        "bg": bgs[0] if len(set(bgs)) == 1 else None,
        "out_dim_range": (min(d[0] for d in out_dims), max(d[0] for d in out_dims),
                          min(d[1] for d in out_dims), max(d[1] for d in out_dims)),
    }


def _predicted_dims(inv, test_input):
    kind, val = inv["dim_rule"]
    ih, iw = _dims(test_input)
    if kind == "same":
        return (ih, iw)
    if kind == "scale":
        return (ih * val[0], iw * val[1])
    if kind == "const":
        return val
    return None


# ---------- the cheap verifier: consistency VIOLATION energy (lower = better) ----------
def _violation_features(cand, inv, test_input):
    ch, cw = _dims(cand)
    pred = _predicted_dims(inv, test_input)
    dim_v = 0.0 if pred is None else (0.0 if (ch, cw) == pred else 1.0)
    cc = _colors(cand)
    novel = cc - inv["all_palette"]
    palette_v = (len(novel) / len(cc)) if cc else 1.0
    bg_v = 0.0 if inv["bg"] is None else (0.0 if _bg(cand) == inv["bg"] else 1.0)
    lo_h, hi_h, lo_w, hi_w = inv["out_dim_range"]
    # size sanity: candidate dims within [0.5x, 2x] of the train output dim range
    size_v = 0.0 if (ch and cw and 1 <= ch <= 30 and 1 <= cw <= 30
                     and 0.5 * lo_h <= ch <= 2 * hi_h and 0.5 * lo_w <= cw <= 2 * hi_w) else 1.0
    return {"dim": dim_v, "palette": palette_v, "bg": bg_v, "size": size_v}


def _energy(feats):
    return sum(feats.values()) / len(feats)


# ---------- distractor generators (deterministic per task+seed; no LLM) ----------
def _rand_grid(h, w, rng, palette):
    pal = sorted(palette) or [0]
    return [[rng.choice(pal) for _ in range(w)] for _ in range(h)]


def _perturb(gold, rng, frac=0.12):
    g = [row[:] for row in gold]
    h, w = _dims(g)
    pal = sorted(_colors(g)) or [0]
    n = max(1, int(frac * h * w))
    for _ in range(n):
        r, c = rng.randrange(h), rng.randrange(w)
        g[r][c] = rng.choice(pal)
    return g


def _color_swap(gold, rng):
    cols = sorted(_colors(gold))
    if len(cols) < 2:
        return None
    a, b = rng.sample(cols, 2)
    return [[b if x == a else a if x == b else x for x in row] for row in gold]


def _wrong_dim(gold, rng):
    h, w = _dims(gold)
    nh, nw = max(1, h + rng.choice([-1, 1, 2])), max(1, w + rng.choice([-1, 1, 2]))
    if (nh, nw) == (h, w):
        nh += 1
    pal = sorted(_colors(gold)) or [0]
    return [[(gold[r][c] if r < h and c < w else pal[0]) for c in range(nw)] for r in range(nh)]


def _transpose(gold):
    h, w = _dims(gold)
    return [[gold[r][c] for r in range(h)] for c in range(w)]


def _distractors(gold, test_input, all_golds, rng):
    easy, hard = {}, {}
    easy["copy_input"] = [row[:] for row in test_input]
    gh, gw = _dims(gold)
    wt = all_golds[rng.randrange(len(all_golds))]
    easy["wrong_task_gold"] = wt
    easy["random"] = _rand_grid(gh, gw, rng, set(range(10)))
    easy["blank"] = [[0] * gw for _ in range(gh)]
    hard["perturbed_gold"] = _perturb(gold, rng)
    sw = _color_swap(gold, rng)
    if sw is not None:
        hard["color_swap_gold"] = sw
    hard["wrong_dim_gold"] = _wrong_dim(gold, rng)
    tp = _transpose(gold)
    if _dims(tp) != _dims(gold) or tp != gold:
        hard["transposed_gold"] = tp

    def _ne(g):  # drop any distractor accidentally equal to gold
        return g is not None and g != gold
    easy = {k: v for k, v in easy.items() if _ne(v)}
    hard = {k: v for k, v in hard.items() if _ne(v)}
    return easy, hard


def _auroc(pos, neg):
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def run(split="training", limit=None, seed=0, write=True):
    rng = random.Random(seed)
    ch = json.load(open(ARC / f"arc-agi_{split}_challenges.json"))
    so = json.load(open(ARC / f"arc-agi_{split}_solutions.json"))
    task_ids = list(ch)
    if limit:
        task_ids = task_ids[:limit]
    all_golds = [so[t][0] for t in task_ids if so.get(t)]

    easy_top1 = hard_top1 = both_top1 = n_eval = 0
    gold_e, easy_e, hard_e = [], [], []
    feat_gold = {k: [] for k in ("dim", "palette", "bg", "size")}
    feat_distr = {k: [] for k in feat_gold}
    dim_rule_dist = Counter()
    per_distr = {}  # distractor_kind -> list of (gold_energy, distractor_energy)

    for t in task_ids:
        task = ch[t]
        inv = _build_invariants(task["train"])
        dim_rule_dist[inv["dim_rule"][0]] += 1
        for ti, test in enumerate(task["test"]):
            if not so.get(t) or ti >= len(so[t]):
                continue
            gold = so[t][ti]
            tin = test["input"]
            easy, hard = _distractors(gold, tin, all_golds, rng)
            if not easy and not hard:
                continue
            n_eval += 1
            gf = _violation_features(gold, inv, tin)
            ge = _energy(gf)
            gold_e.append(-ge)  # negate so higher score = better (for AUROC)
            for k in feat_gold:
                feat_gold[k].append(gf[k])
            ee = [_energy(_violation_features(d, inv, tin)) for d in easy.values()]
            he = [_energy(_violation_features(d, inv, tin)) for d in hard.values()]
            for kind, d in list(easy.items()) + list(hard.items()):
                df = _violation_features(d, inv, tin)
                for k in feat_distr:
                    feat_distr[k].append(df[k])
                per_distr.setdefault(kind, []).append((ge, _energy(df)))
            easy_e += [-e for e in ee]
            hard_e += [-e for e in he]
            # top-1: gold strictly-lowest energy among gold+distractors (ties = fail)
            if ee and ge < min(ee):
                easy_top1 += 1
            if he and ge < min(he):
                hard_top1 += 1
            alld = ee + he
            if alld and ge < min(alld):
                both_top1 += 1

    # per-feature discrimination AUROC (gold violation vs distractor violation; lower
    # violation should mean gold -> use -violation as score)
    feat_auroc = {k: _auroc([-x for x in feat_gold[k]], [-x for x in feat_distr[k]])
                  for k in feat_gold}
    # per-distractor pairwise win-rate: fraction where gold has STRICTLY lower energy than
    # this distractor kind (ties excluded from the win count -> honest "can it separate?")
    per_distr_sep = {}
    for kind, pairs in sorted(per_distr.items()):
        wins = sum(1 for ge, de in pairs if ge < de)
        ties = sum(1 for ge, de in pairs if ge == de)
        per_distr_sep[kind] = {"n": len(pairs), "gold_strictly_better_rate": round(wins / len(pairs), 4),
                               "tie_rate": round(ties / len(pairs), 4)}

    res = {
        "n_eval_test_inputs": n_eval,
        "easy_distractor_top1_acc": round(easy_top1 / n_eval, 4) if n_eval else None,
        "hard_distractor_top1_acc": round(hard_top1 / n_eval, 4) if n_eval else None,
        "combined_top1_acc": round(both_top1 / n_eval, 4) if n_eval else None,
        "easy_discrimination_auroc": _auroc(gold_e, easy_e) and round(_auroc(gold_e, easy_e), 4),
        "hard_discrimination_auroc": _auroc(gold_e, hard_e) and round(_auroc(gold_e, hard_e), 4),
        "per_feature_auroc": {k: (round(v, 4) if v is not None else None)
                              for k, v in feat_auroc.items()},
        "per_distractor_separation": per_distr_sep,
        "dim_rule_coverage": dict(dim_rule_dist),
    }
    easy_a = res["easy_discrimination_auroc"]
    hard_a = res["hard_discrimination_auroc"]
    # HONEST taxonomy: separation strength (AUROC) x selection ability (top1).
    if easy_a is None:
        finding = "undefined"
    elif easy_a >= 0.85:
        finding = "STRONG_structural_pruner"
    elif easy_a >= 0.70:
        finding = "MODERATE_structural_pruner_dim_dominated"
    else:
        finding = "WEAK_even_on_easy"
    # selection: can it pick gold uniquely? (top1). Near-miss ties make this ~0 => PRUNER_NOT_SELECTOR.
    select = ("can_select" if (res["combined_top1_acc"] or 0) >= 0.30 else "PRUNER_NOT_SELECTOR")
    verdict = (f"complete: arc_grid_verifier_{finding}_{select}"
               f"_easyAUROC{easy_a}_hardAUROC{hard_a}_combined_top1{res['combined_top1_acc']}"
               f"_n{n_eval}")
    art = {
        "experiment": "arc_grid_verifier_discriminator_draft",
        "title": "arc_grid_cheap_verifier_as_pruner",
        "honest_verdict": verdict,
        "inference_substrate": "rule_based_grid_verifier_against_cached_arc_solutions",
        "domain": "arc_agi1_grid",
        "split": split, "n_tasks": len(task_ids), "random_seed": seed,
        "label_source": "gold_arc_solutions",
        "results": res,
        "gate": "easy discrimination AUROC >= 0.70 (cheap pruning works) AND hard < 0.65 "
                "(induction needed -> verifier prunes, generator induces)",
        "no_llm_used": True, "no_induction": True,
        "interpretation": (
            "The cheap verifier checks dimension/palette/background CONSISTENCY from train "
            "pairs only -- it does NOT induce the rule. If it ranks gold above EASY distractors "
            "(dimension/palette-wrong) it earns the harness PRUNER role (fewer generator actions "
            "= the ARC-AGI-3 efficiency axis). If it cannot separate HARD distractors (perturbed "
            "gold) that is the EXPECTED ceiling: discriminating a nearly-right grid needs "
            "rule-induction, which is the generator's job. This is the ARC instantiation of "
            "verifier-prunes / generator-induces (cf. math SC-bound, code execution-bound)."
        ),
        "principle_label_source": "gold ARC solutions are the oracle; distractors are cheap "
                                  "deterministic perturbations -> no LLM, no induction, honest "
                                  "test of the PRUNER role only",
    }
    if write:
        OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


if __name__ == "__main__":
    a = run()
    print(f"-> {a['honest_verdict']}")
    r = a["results"]
    print(f"   n={r['n_eval_test_inputs']} dim_rule_coverage={r['dim_rule_coverage']}")
    print(f"   EASY: top1={r['easy_distractor_top1_acc']} auroc={r['easy_discrimination_auroc']}")
    print(f"   HARD: top1={r['hard_distractor_top1_acc']} auroc={r['hard_discrimination_auroc']}")
    print(f"   combined_top1={r['combined_top1_acc']}")
    print(f"   per_feature_auroc={r['per_feature_auroc']}")
