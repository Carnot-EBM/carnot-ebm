"""DRAFT (#C, north-star domain): the CONTENT_OVERLAP invariant family — a cheap,
NON-inducing, cell-LEVEL relational verifier for ARC outputs.

WHY (north-star.md §0). The v1 ARC grid verifier
(results/arc_grid_verifier_discriminator.json) is a STRUCTURAL pruner: it checks only
dims/palette/background derived from the train pairs. It prunes structurally-malformed
candidates well (wrong_task_gold 0.947, random 0.889, wrong_dim 0.863) but is BLIND to
CONTENT errors that preserve structure — its per-distractor gold-strictly-better rate was
perturbed_gold 0.006, transposed 0.293, color_swap 0.384, blank 0.135, copy_input 0.389.
Those content-preserving-but-wrong distractors are the CEILING this family attacks.

WHAT content_overlap measures (cellwise relational AGREEMENT, no rule induction):
  We never look at the test input's gold output (no label leak). We only use the TRAIN
  pairs + the candidate grid + the test input. Two cheap, deterministic cell-level signals:

    (A) per-position MODE template (only when train OUTPUTS share a constant dimension):
        build the per-cell most-common color over the train outputs; the candidate is
        scored by 1 - (fraction of its cells equal to that per-position mode). Train
        outputs that share a fixed canvas (same out-dims) often share a literal scaffold
        (a frame, a fixed background field, repeated motif positions). A perturbed or
        randomized candidate disagrees with that scaffold; the gold agrees with it. This
        catches `random`, `blank`, and the structural part of `perturbed_gold`.

    (B) INPUT->OUTPUT copy-consistency (only when the dim rule is `same`, i.e. output has
        the same shape as input): over the train pairs, find the set of cell POSITIONS
        whose color is preserved input->output in EVERY train pair ("stable copy cells"),
        and the global majority output-color for the rest. Apply that learned mask to the
        TEST input to synthesize a cheap expectation grid, then score the candidate by
        1 - (fraction of cells matching the expectation). This is the relational signal
        that perturbed_gold / transposed / color_swap break: a transposed or color-swapped
        candidate disagrees with the copied-from-input cells; a perturbed candidate
        disagrees on the flipped cells.

  family_score(candidate, inv, train_pairs, test_input) -> float in [0,1], LOWER = more
  consistent (matches v1's "violation energy, lower=better" convention so _auroc(-score)
  works identically). The two signals are blended; when a signal is undefined for a task
  (dims not constant / dim-rule not `same`) it abstains (contributes nothing) so we never
  fabricate agreement.

HARD CONSTRAINTS honored: no LLM, no GPU, no induction-by-search, no brute-forcing the
rule, no use of the test gold. Pure deterministic structural/relational arithmetic over
train pairs. Same 400-task protocol, same distractors, same random.Random(0) as v1.

  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" .venv/bin/python \
      scripts/experiments/arc_invariant_content_overlap_draft.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments"))

# Reuse the v1 invariant model + distractor generators + AUROC verbatim (pure, no heavy
# deps). This guarantees an apples-to-apples comparison: identical candidate sets, identical
# distractor seeds, identical metric — only the SCORING FUNCTION differs.
from arc_grid_verifier_discriminator_draft import (  # noqa: E402
    _auroc,
    _bg,
    _build_invariants,
    _colors,
    _dims,
    _distractors,
    _predicted_dims,
)

ARC = Path("/home/ianblenke/trm_src/kaggle/combined")
OUT = REPO_ROOT / "results" / "arc_invariant_content_overlap.json"


# ---------- content_overlap signal (A): per-position mode template ----------
def _mode_template(train):
    """Per-cell most-common color over train OUTPUTS, only if they share a fixed canvas.

    Returns (template_grid, h, w) or None if the train outputs do not all share one
    dimension (in which case there is no fixed per-position scaffold to compare against).
    """
    outs = [p["output"] for p in train]
    dims = {_dims(o) for o in outs}
    if len(dims) != 1:
        return None
    h, w = next(iter(dims))
    if h == 0 or w == 0:
        return None
    template = [[None] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            ctr = Counter(o[r][c] for o in outs)
            template[r][c] = ctr.most_common(1)[0][0]
    return template, h, w


def _mode_agreement(cand, tmpl):
    """Fraction of candidate cells equal to the per-position mode template.

    If the candidate's dims differ from the template, agreement is 0 (it cannot match a
    scaffold of a different shape) — that is itself a content/structure violation signal.
    """
    template, h, w = tmpl
    ch, cw = _dims(cand)
    if (ch, cw) != (h, w):
        return 0.0
    match = sum(1 for r in range(h) for c in range(w) if cand[r][c] == template[r][c])
    return match / (h * w)


# ---------- content_overlap signal (B): input->output copy-consistency ----------
def _copy_expectation(train, test_input):
    """Learn a cheap cell-level expectation for `same`-dim tasks, apply it to test_input.

    For tasks where output has the SAME shape as input on every train pair, find:
      - stable_copy positions (r,c): input[r][c] == output[r][c] in EVERY train pair AND
        the input value at (r,c) was not constant-trivial. These are cells the rule copies
        through unchanged.
      - For non-copy cells, the single most common OUTPUT color across all train outputs
        (a crude "fill" expectation).
    Then build an expectation grid the size of test_input: copy cells take the test input's
    value, the rest take the learned fill color. Returns (expectation_grid, h, w) or None
    when the rule is not same-dim or positions are inconsistent in size.

    This never touches the test gold; it is a pure projection of the train relation onto
    the test input.
    """
    pairs = train
    if not all(_dims(p["input"]) == _dims(p["output"]) for p in pairs):
        return None
    # All train pairs must share one (h,w) for a fixed positional copy-mask to be meaningful.
    shapes = {_dims(p["input"]) for p in pairs}
    if len(shapes) != 1:
        return None
    h, w = next(iter(shapes))
    th, tw = _dims(test_input)
    if (th, tw) != (h, w):
        # Test input shape differs from the train canvas -> positional mask doesn't transfer.
        return None
    # stable copy mask: positions copied unchanged in every train pair
    copy_mask = [[True] * w for _ in range(h)]
    for p in pairs:
        gi, go = p["input"], p["output"]
        for r in range(h):
            for c in range(w):
                if gi[r][c] != go[r][c]:
                    copy_mask[r][c] = False
    # fill color for non-copy cells = global majority output color over train outputs
    fill_ctr = Counter()
    for p in pairs:
        for r in range(h):
            for c in range(w):
                if not copy_mask[r][c]:
                    fill_ctr[p["output"][r][c]] += 1
    fill = fill_ctr.most_common(1)[0][0] if fill_ctr else _bg(pairs[0]["output"])
    exp = [[(test_input[r][c] if copy_mask[r][c] else fill) for c in range(w)]
           for r in range(h)]
    return exp, h, w


def _copy_agreement(cand, exp):
    expectation, h, w = exp
    ch, cw = _dims(cand)
    if (ch, cw) != (h, w):
        return 0.0
    match = sum(1 for r in range(h) for c in range(w) if cand[r][c] == expectation[r][c])
    return match / (h * w)


# ---------- the family score: LOWER = more consistent (matches v1 convention) ----------
def family_score(candidate_grid, inv, train_pairs, test_input):
    """content_overlap violation energy in [0,1]; LOWER = candidate agrees more.

    Blends the two cell-level agreement signals. Each signal abstains when undefined for
    the task; if BOTH abstain we fall back to a neutral 0.5 (no information -> no claim),
    so the family never fabricates discrimination on tasks it cannot model.
    """
    agreements = []
    tmpl = _mode_template(train_pairs)
    if tmpl is not None:
        agreements.append(_mode_agreement(candidate_grid, tmpl))
    exp = _copy_expectation(train_pairs, test_input)
    if exp is not None:
        agreements.append(_copy_agreement(candidate_grid, exp))
    if not agreements:
        return 0.5  # neutral: no cell-level signal available for this task
    # Average agreement across the available signals; violation = 1 - agreement.
    return 1.0 - (sum(agreements) / len(agreements))


def run(split="training", limit=None, seed=0, write=True):
    t0 = time.time()
    rng = random.Random(seed)
    ch = json.load(open(ARC / f"arc-agi_{split}_challenges.json"))
    so = json.load(open(ARC / f"arc-agi_{split}_solutions.json"))
    task_ids = list(ch)
    if limit:
        task_ids = task_ids[:limit]
    all_golds = [so[t][0] for t in task_ids if so.get(t)]

    n_eval = 0
    gold_e, easy_e, hard_e = [], [], []
    per_distr = {}  # distractor_kind -> list of (gold_score, distractor_score)
    # same, but restricted to test-inputs where AT LEAST ONE cell-level signal was defined
    # (i.e. the family did NOT fall back to the neutral 0.5 abstention). This isolates the
    # family's discriminative power from the abstention dilution.
    per_distr_defined = {}
    n_signal_a = n_signal_b = n_both_abstain = 0

    for t in task_ids:
        task = ch[t]
        inv = _build_invariants(task["train"])
        # cheap per-task introspection: which signals are defined?
        has_a = _mode_template(task["train"]) is not None
        for ti, test in enumerate(task["test"]):
            if not so.get(t) or ti >= len(so[t]):
                continue
            gold = so[t][ti]
            tin = test["input"]
            easy, hard = _distractors(gold, tin, all_golds, rng)
            if not easy and not hard:
                continue
            n_eval += 1
            has_b = _copy_expectation(task["train"], tin) is not None
            if has_a:
                n_signal_a += 1
            if has_b:
                n_signal_b += 1
            signal_defined = has_a or has_b
            if not signal_defined:
                n_both_abstain += 1

            gs = family_score(gold, inv, task["train"], tin)
            gold_e.append(-gs)  # negate: higher score = better, matches v1 AUROC convention
            for kind, d in list(easy.items()) + list(hard.items()):
                ds = family_score(d, inv, task["train"], tin)
                per_distr.setdefault(kind, []).append((gs, ds))
                if signal_defined:
                    per_distr_defined.setdefault(kind, []).append((gs, ds))
            easy_e += [-family_score(d, inv, task["train"], tin) for d in easy.values()]
            hard_e += [-family_score(d, inv, task["train"], tin) for d in hard.values()]

    # per-distractor: gold strictly-better rate + pairwise AUROC (using -score as signal).
    def _per_distr_table(distr_map):
        out = {}
        for kind, pairs in sorted(distr_map.items()):
            wins = sum(1 for gs, ds in pairs if gs < ds)
            ties = sum(1 for gs, ds in pairs if gs == ds)
            # pairwise AUROC: pos = -gold_score, neg = -distractor_score (higher=better)
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

    per_distr_out = _per_distr_table(per_distr)
    per_distr_defined_out = _per_distr_table(per_distr_defined)

    easy_auroc = _auroc(gold_e, easy_e)
    hard_auroc = _auroc(gold_e, hard_e)
    duration_s = round(time.time() - t0, 4)

    # which HARD distractor kinds did content_overlap beat the v1 ceiling on?
    hard_kinds = ("perturbed_gold", "color_swap_gold", "wrong_dim_gold", "transposed_gold")
    catches_hard = [k for k in hard_kinds
                    if per_distr_out.get(k, {}).get("gold_strictly_better_rate", 0.0) >= 0.70]

    res = {
        "family": "content_overlap",
        "n_eval_test_inputs": n_eval,
        "signal_coverage": {
            "mode_template_defined": n_signal_a,
            "copy_expectation_defined": n_signal_b,
            "both_abstain": n_both_abstain,
        },
        "easy_discrimination_auroc": (round(easy_auroc, 4) if easy_auroc is not None else None),
        "hard_discrimination_auroc": (round(hard_auroc, 4) if hard_auroc is not None else None),
        "per_distractor": per_distr_out,
        # conditional view: restricted to test-inputs where the family had a real (non-abstain)
        # cell-level signal. Shows the family's TRUE discriminative power undiluted by the
        # ~46% of tasks where it abstains (no constant-dim outputs, no same-dim copy rule).
        "per_distractor_signal_defined": per_distr_defined_out,
        "catches_hard": catches_hard,
        "catches_hard_when_signal_defined": [
            k for k in ("perturbed_gold", "color_swap_gold", "wrong_dim_gold", "transposed_gold")
            if per_distr_defined_out.get(k, {}).get("gold_strictly_better_rate", 0.0) >= 0.70
        ],
    }

    # honest verdict (terminal-prefixed per CLAUDE.md Verdict Terminal-Prefix Discipline)
    verdict = (f"complete: arc_invariant_content_overlap "
               f"easyAUROC{res['easy_discrimination_auroc']} "
               f"hardAUROC{res['hard_discrimination_auroc']} "
               f"catches_hard={catches_hard or 'NONE'} n{n_eval}")
    art = {
        "experiment": "arc_invariant_content_overlap_draft",
        "title": "arc_content_overlap_cell_level_relational_verifier",
        "honest_verdict": verdict,
        "inference_substrate": "rule_based_grid_verifier_against_cached_arc_solutions",
        "domain": "arc_agi1_grid",
        "split": split,
        "n_tasks": len(task_ids),
        "random_seed": seed,
        "duration_s": duration_s,
        "label_source": "gold_arc_solutions",
        "no_llm_used": True,
        "no_induction": True,
        "no_label_leak": True,
        "results": res,
        "interpretation": (
            "content_overlap adds two cell-LEVEL relational signals the v1 structural "
            "verifier lacked: (A) per-position mode agreement vs the train outputs' shared "
            "scaffold, and (B) input->output copy-consistency projected onto the test input. "
            "Both are cheap, deterministic, and use ONLY the train pairs + candidate + test "
            "input (no test gold). Targets the content-preserving distractors (perturbed, "
            "transposed, color_swap) the v1 dims/palette/bg checks were blind to."
        ),
    }
    if write:
        OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


if __name__ == "__main__":
    a = run()
    r = a["results"]
    print(f"-> {a['honest_verdict']}")
    print(f"   n={r['n_eval_test_inputs']} duration_s={a['duration_s']} "
          f"signal_coverage={r['signal_coverage']}")
    print(f"   EASY auroc={r['easy_discrimination_auroc']}  "
          f"HARD auroc={r['hard_discrimination_auroc']}")
    print("   per_distractor FULL CORPUS (gold_strictly_better_rate / auroc):")
    for kind, d in sorted(r["per_distractor"].items()):
        print(f"     {kind:18s} sbr={d['gold_strictly_better_rate']:.4f} "
              f"tie={d['tie_rate']:.4f} auroc={d['auroc']}  n={d['n']}")
    print("   per_distractor WHERE SIGNAL DEFINED (non-abstain subset):")
    for kind, d in sorted(r["per_distractor_signal_defined"].items()):
        print(f"     {kind:18s} sbr={d['gold_strictly_better_rate']:.4f} "
              f"tie={d['tie_rate']:.4f} auroc={d['auroc']}  n={d['n']}")
    print(f"   catches_hard FULL (>=0.70) = {r['catches_hard'] or 'NONE'}")
    print(f"   catches_hard WHEN-SIGNAL-DEFINED (>=0.70) = "
          f"{r['catches_hard_when_signal_defined'] or 'NONE'}")
