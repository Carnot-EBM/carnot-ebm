"""
Scissor-plot STAGE 0 — necessary-condition cut (no GPU), per the design doc
docs/research-notes/verifier-moat-scissor-plot-design.md (revised after the Deep
Think methodology check, verdict FIX-FIRST).

THE QUESTION (necessary condition for the verifier moat): on the RESIDUAL (answers a
clean-syntax model got WRONG), does Carnot's mechanistically-INDEPENDENT constraint
verifier (Z3 arithmetic, the probe-ablated pure-constraint core DT recommended) catch
errors that the cheap self-verification baseline (SELF-CONSISTENCY = majority vote over
samples) WAVES THROUGH? If it does not beat SC at all even here — on a clean-syntax,
difficulty-controlled residual — the moat is absent at this point and the GPU sweep is
not worth running (DT Q3: this corpus is a clean mid-model, NOT 0.5B babble, so a flat
gap is about LOGIC, not formatting).

DATA: data/p01_difficulty_matched_generations.jsonl — 93 MATH L3/L4 problems from ONE
clean model; greedy answer + correctness label + K=6 samples (for SC) + level.

METRIC (DT-corrected):
- Verifier "likely-correct" score per problem:
    constraint: c = 1 - Z3.score(text)  (1=no arithmetic violation; 0=violations);
                ABSTAIN when no checkable arithmetic (Z3 sentinel 0.5).
    self-consistency: agreement share of greedy's answer among the K sampled answers
                (high = samples agree with greedy = SC confident greedy is right).
- Q2 coverage/operating-point match: compare at a MATCHED recall-on-CORRECT (accept-rate
  on the 32 correct answers). At that point, FPR-on-residual = accept-rate on the WRONG
  answers (a false-accept of a wrong answer). Lower = better.
- Q4 stratify by difficulty level (report L3, the bulk).
- Report coverage (constraint only fires on arithmetic-bearing trajectories).

SCOPE / honesty: Z3 constraint core only (not the full 15-verifier conformal ensemble,
which needs cached per-verifier score artifacts); ONE clean model (the mid bracket
point), SC baseline (logprob is null in cache). This is the NECESSARY-CONDITION cut, not
the full scissor. A clean negative here = stop; a clear edge = build the GPU sweep.
"""
import os
import sys
import json
import hashlib
import time

PROJECT_ROOT = "/home/ianblenke/github.com/ianblenke/carnot"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "python"))

from carnot.verify.z3_math_verifier import Z3MathVerifier
import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


def load_corpus():
    rows = [json.loads(l) for l in open(os.path.join(PROJECT_ROOT, "data/p01_difficulty_matched_generations.jsonl"))]
    return rows


def sc_agreement(greedy_ans, sampled_answers):
    """Share of the K samples whose extracted answer equals greedy's answer.
    High => self-consistent => SC is confident greedy is right (the cheap baseline)."""
    vals = [a for a in (sampled_answers or [])]
    if not vals:
        return None
    match = sum(1 for a in vals if a is not None and a == greedy_ans)
    return match / len(vals)


def matched_fpr(scores, labels, abstain, target_recall_on_correct):
    """At the threshold giving `target_recall_on_correct` (accept-rate on label==1),
    return accept-rate on label==0 (FPR-on-residual), computed on NON-abstained items.
    scores: higher = more-likely-correct (accept)."""
    idx = [i for i in range(len(scores)) if not abstain[i] and scores[i] is not None]
    s = np.array([scores[i] for i in idx]); y = np.array([labels[i] for i in idx])
    pos = s[y == 1]; neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None, None, len(idx)
    # threshold = the value s.t. we accept target_recall_on_correct of the correct ones
    thr = np.quantile(pos, 1.0 - target_recall_on_correct)   # accept score >= thr
    recall_correct = float((pos >= thr).mean())
    fpr_residual = float((neg >= thr).mean())                 # wrong answers accepted
    return fpr_residual, recall_correct, len(idx)


def auroc(scores, labels, abstain):
    if roc_auc_score is None:
        return None
    idx = [i for i in range(len(scores)) if not abstain[i] and scores[i] is not None]
    y = [labels[i] for i in idx]; s = [scores[i] for i in idx]
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def main():
    t0 = time.time()
    rows = load_corpus()
    z3 = Z3MathVerifier()

    labels, levels = [], []
    c_scores, c_abstain = [], []      # constraint (Z3)
    sc_scores, sc_abstain = [], []    # self-consistency baseline
    for r in rows:
        g = r["greedy"]
        label = 1 if (g.get("correct") is True or r.get("greedy_correct") is True) else 0
        labels.append(label); levels.append(r.get("level"))
        # constraint: Z3 arithmetic violation on the greedy trajectory
        v = z3.score(g.get("text") or "")            # 0=clean, 1=all violated, 0.5=no arithmetic
        if abs(v - 0.5) < 1e-9:
            c_scores.append(None); c_abstain.append(True)     # no checkable arithmetic -> abstain
        else:
            c_scores.append(1.0 - v); c_abstain.append(False) # higher = cleaner = likely correct
        # self-consistency baseline (never abstains)
        sc = sc_agreement(g.get("extracted_answer_norm"), r.get("sampled_answers"))
        sc_scores.append(sc); sc_abstain.append(sc is None)

    n = len(rows); n_res = sum(1 for x in labels if x == 0)
    cov_constraint = sum(1 for a in c_abstain if not a) / n

    out = {"n_problems": n, "n_residual": n_res,
           "constraint_coverage": round(cov_constraint, 3),
           "auroc_constraint_z3": auroc(c_scores, labels, c_abstain),
           "auroc_self_consistency": auroc(sc_scores, labels, sc_abstain),
           "matched_operating_points": {}}

    # Q2: at matched recall-on-correct, compare FPR-on-residual (constraint vs SC)
    for rec in (0.70, 0.80, 0.90):
        cf, cr, cn = matched_fpr(c_scores, labels, c_abstain, rec)
        sf, sr, sn = matched_fpr(sc_scores, labels, sc_abstain, rec)
        out["matched_operating_points"][f"recall_on_correct={rec}"] = {
            "constraint_fpr_residual": None if cf is None else round(cf, 3),
            "self_consistency_fpr_residual": None if sf is None else round(sf, 3),
            "constraint_beats_sc": (cf is not None and sf is not None and cf < sf),
            "constraint_covered_n": cn, "sc_covered_n": sn,
        }

    # Q4: stratify by level (L3 is the bulk)
    out["by_level"] = {}
    for lv in sorted(set(levels)):
        ix = [i for i in range(n) if levels[i] == lv]
        if len(ix) < 8:
            out["by_level"][str(lv)] = {"n": len(ix), "note": "too small to stratify"}
            continue
        cs = [c_scores[i] for i in ix]; ca = [c_abstain[i] for i in ix]
        ss = [sc_scores[i] for i in ix]; sa = [sc_abstain[i] for i in ix]; ly = [labels[i] for i in ix]
        cf, _, _ = matched_fpr(cs, ly, ca, 0.80); sf, _, _ = matched_fpr(ss, ly, sa, 0.80)
        out["by_level"][str(lv)] = {
            "n": len(ix), "n_residual": sum(1 for x in ly if x == 0),
            "auroc_constraint": auroc(cs, ly, ca), "auroc_sc": auroc(ss, ly, sa),
            "constraint_fpr@rec0.8": None if cf is None else round(cf, 3),
            "sc_fpr@rec0.8": None if sf is None else round(sf, 3),
        }

    # honest verdict
    mp = out["matched_operating_points"].get("recall_on_correct=0.8", {})
    beats = mp.get("constraint_beats_sc")
    au_c, au_s = out["auroc_constraint_z3"], out["auroc_self_consistency"]
    if beats and au_c and au_c > 0.55:
        verdict = ("complete: scissor_stage0_NECESSARY_CONDITION_MET_constraint_beats_self_consistency_on_residual"
                   f"_fpr_{mp.get('constraint_fpr_residual')}_vs_{mp.get('self_consistency_fpr_residual')}_auroc_{au_c:.3f}_BUILD_GPU_SWEEP")
    elif au_c is None or out["constraint_coverage"] < 0.15:
        verdict = (f"complete: scissor_stage0_INCONCLUSIVE_constraint_coverage_{out['constraint_coverage']}_too_low"
                   "_z3_rarely_fires_on_this_corpus_need_broader_constraint_set_or_arithmetic_heavy_corpus")
    else:
        verdict = ("complete: scissor_stage0_NECESSARY_CONDITION_NOT_MET_constraint_does_not_beat_self_consistency_on_residual"
                   f"_constraint_fpr_{mp.get('constraint_fpr_residual')}_sc_fpr_{mp.get('self_consistency_fpr_residual')}"
                   f"_auroc_c_{au_c}_s_{au_s}_DO_NOT_BUILD_SWEEP_moat_questionable")

    art = {
        "experiment": "scissor_stage0_necessary_condition",
        "honest_verdict": verdict,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "scope": ("Z3 constraint core only (probe-ablated pure-constraint, not full conformal ensemble); "
                  "ONE clean mid-model; SC baseline (cached logprob is null); 93 MATH L3/L4 problems. "
                  "NECESSARY-CONDITION cut per the scissor-plot design doc, NOT the full scissor."),
        **out,
        "model_specs": {"verifier": "Z3MathVerifier (arithmetic constraint)", "baseline": "self_consistency_maj@6"},
        "random_seed": 0,
        "reproducibility_checksum": hashlib.sha256(
            json.dumps({"corpus": "p01_difficulty_matched", "n": n}, sort_keys=True).encode()).hexdigest(),
        "duration_s": round(time.time() - t0, 3),
    }
    path = os.path.join(PROJECT_ROOT, "results", "scissor_stage0_necessary_condition.json")
    with open(path, "w") as f:
        json.dump(art, f, indent=2)
    print(json.dumps(out, indent=2))
    print("\n" + verdict)
    print("->", path)


if __name__ == "__main__":
    main()
