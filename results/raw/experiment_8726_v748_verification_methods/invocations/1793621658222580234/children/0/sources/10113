"""DRAFT: can we LEARN an outcome-verifier from gold that beats the fixed process-reward?

Tonight's on-policy result: the fixed process-verifier FAILED as a self-improvement reward
(process_weighted ~ base) because it scores LOCAL step-validity (process), not OUTCOME
correctness (process != outcome; raw fraction-certified ranks trace-outcome at AUROC 0.73).
The proposed rescue: TRAIN an outcome-verifier on (trace, gold-outcome) -- the same gold
oracle that taught the generator -- so it certifies OUTCOMES.

This de-risks that BEFORE the full RFT rebuild (cheap, CPU, runs concurrent with the GPU
3-seed). Extract MULTIPLE per-trace verifier features, train a logistic on gold-outcome with
a HELD-OUT split, and ask: does the learned multi-feature outcome-verifier beat the single
fraction-certified feature (0.73)? If yes (>= ~0.80 held-out), a learned verifier is worth
swapping into the process-reward arm. If ~0.73, the verifier features don't carry more
outcome signal -> need richer features (self-consistency, answer-chain check) first.

  .venv/bin/python scripts/experiments/outcome_verifier_learn_derisk_draft.py
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path

from carnot.eval.verifier_error_independence_scissor_at_scale import (
    FoVerPanel, score_carnot_ensemble,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GSM8K = REPO_ROOT / "data" / "p01_gsm8k_generations.jsonl"
OUT = REPO_ROOT / "results" / "outcome_verifier_learn_derisk.json"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _chunks(text):
    body = _THINK.sub("", str(text)).strip()
    return [s.strip() for s in re.split(r"\n\s*\n", body)
            if len(s.strip()) >= 12 and re.search(r"[a-zA-Z0-9]", s)]


def _extract(t):
    n = _NUM.findall(str(t).replace(",", ""))
    return n[-1] if n else None


def _load_traces():
    traces = []
    with GSM8K.open() as f:
        for line in f:
            r = json.loads(line)
            gold = str(r.get("gold") or "").strip()
            # collect this question's answers first (for self-consistency)
            grp = []
            for s in (r.get("samples") or []):
                txt = s if isinstance(s, str) else str(s.get("text") or "")
                if not txt.strip():
                    continue
                ans = str((s.get("answer") if isinstance(s, dict) else None) or _extract(txt)).strip()
                grp.append((txt, ans))
            answers = [a for _, a in grp]
            for txt, ans in grp:
                # SELF-CONSISTENCY feature: fraction of THIS question's samples sharing
                # this trace's answer (majority-agreement; NO gold used).
                sc = answers.count(ans) / len(answers) if answers else 0.0
                traces.append({"text": txt, "gold_correct": int(ans == gold), "sc_agreement": sc})
    return traces


def _auroc(labels, scores):
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _fit_logreg(x, y, iters=600, lr=0.1):
    nf = len(x[0])
    w = [0.0] * (nf + 1)
    n = len(x)
    # standardize
    means = [sum(r[j] for r in x) / n for j in range(nf)]
    stds = [(sum((r[j] - means[j]) ** 2 for r in x) / n) ** 0.5 or 1.0 for j in range(nf)]
    xs = [[(r[j] - means[j]) / stds[j] for j in range(nf)] for r in x]
    for _ in range(iters):
        g = [0.0] * (nf + 1)
        for xi, yi in zip(xs, y):
            z = w[-1] + sum(w[j] * xi[j] for j in range(nf))
            p = 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))
            e = p - yi
            for j in range(nf):
                g[j] += e * xi[j]
            g[-1] += e
        for j in range(nf + 1):
            w[j] -= lr * g[j] / n
    return w, means, stds


def _predict(w, means, stds, r):
    nf = len(means)
    xi = [(r[j] - means[j]) / stds[j] for j in range(nf)]
    z = w[-1] + sum(w[j] * xi[j] for j in range(nf))
    return 1.0 / (1.0 + math.exp(-max(-30, min(30, z))))


def run(write=True):
    traces = _load_traces()
    # score every chunk once
    chunk_texts, owner = [], []
    for ti, t in enumerate(traces):
        for c in _chunks(t["text"]):
            chunk_texts.append(c)
            owner.append(ti)
    panel = FoVerPanel(rows=tuple({"idx": i} for i in range(len(chunk_texts))),
                       labels=tuple(0 for _ in chunk_texts), texts=tuple(chunk_texts),
                       panel_sha256=hashlib.sha256("".join(chunk_texts).encode()).hexdigest())
    scoring = score_carnot_ensemble(panel, REPO_ROOT)
    reward = [1.0 - float(s) for s in scoring.scores]
    pcorrect = [1 - int(p) for p in scoring.error_preds]
    by_trace_r, by_trace_p = {}, {}
    for i, ti in enumerate(owner):
        by_trace_r.setdefault(ti, []).append(reward[i])
        by_trace_p.setdefault(ti, []).append(pcorrect[i])

    # per-trace FEATURES (all process-side, NO gold leak into features)
    feats, labels, frac_cert = [], [], []
    for ti, t in enumerate(traces):
        rs = by_trace_r.get(ti, [0.0])
        ps = by_trace_p.get(ti, [0])
        mean_r = sum(rs) / len(rs)
        min_r = min(rs)
        fc = sum(ps) / len(ps)
        std_r = (sum((x - mean_r) ** 2 for x in rs) / len(rs)) ** 0.5
        n_ch = len(rs)
        char_len = len(t["text"])
        sc = float(t.get("sc_agreement", 0.0))  # self-consistency (cheap outcome signal)
        feats.append([mean_r, min_r, fc, std_r, float(n_ch), float(char_len) / 1000.0, sc])
        labels.append(t["gold_correct"])
        frac_cert.append(fc)

    n = len(feats)
    base_rate = sum(labels) / n
    # held-out split (train 70 / test 30)
    idx = list(range(n))
    random.Random(0).shuffle(idx)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    w, means, stds = _fit_logreg([feats[i] for i in tr], [labels[i] for i in tr])
    learned_test = [_predict(w, means, stds, feats[i]) for i in te]
    auroc_learned = _auroc([labels[i] for i in te], learned_test)
    auroc_frac = _auroc([labels[i] for i in te], [frac_cert[i] for i in te])
    # single-feature aurocs (full set, for diagnostics)
    fnames = ["mean_reward", "min_reward", "fraction_certified", "std_reward", "n_chunks",
              "char_len_k", "self_consistency"]
    single = {fnames[j]: round(_auroc(labels, [f[j] for f in feats]) or 0.5, 4) for j in range(len(fnames))}
    # also: process-only learned (no SC) vs process+SC, to isolate SC's contribution
    proc_only = [f[:6] for f in feats]
    wp, mp, sp = _fit_logreg([proc_only[i] for i in tr], [labels[i] for i in tr])
    auroc_proc_only = _auroc([labels[i] for i in te], [_predict(wp, mp, sp, proc_only[i]) for i in te])

    gate = bool(auroc_learned is not None and auroc_learned >= 0.80)
    verdict = (f"complete: outcome_verifier_learn_{'BEATS' if gate else 'no_gain'}"
               f"_learned{round(auroc_learned,3) if auroc_learned else 'na'}"
               f"_vs_fraccert{round(auroc_frac,3) if auroc_frac else 'na'}_base{base_rate:.3f}")
    art = {
        "experiment": "outcome_verifier_learn_derisk_draft",
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "n_traces": n, "gold_correct_base_rate": round(base_rate, 4),
        "heldout_auroc_learned_outcome_verifier": None if auroc_learned is None else round(auroc_learned, 4),
        "heldout_auroc_process_only_no_SC": None if auroc_proc_only is None else round(auroc_proc_only, 4),
        "heldout_auroc_fraction_certified_baseline": None if auroc_frac is None else round(auroc_frac, 4),
        "single_feature_auroc_full": single,
        "gate": "learned held-out outcome-AUROC >= 0.80 (worth swapping into RFT)",
        "gate_pass": gate,
        "interpretation": (
            "If learned >> fraction_certified (~0.73): the gold-trained outcome-verifier carries "
            "real outcome signal the fixed process-reward misses -> swap it into the process-reward "
            "RFT arm and re-test (does 'verifier teaches' turn positive?). If learned ~ fraction_"
            "certified: process features alone don't capture outcome -> add richer features "
            "(self-consistency across K samples, answer-chain arithmetic check) before the RFT build."
        ),
    }
    if write:
        OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


if __name__ == "__main__":
    a = run()
    print(f"-> {a['honest_verdict']}")
    print(f"   n={a['n_traces']} base_rate={a['gold_correct_base_rate']}")
    print(f"   LEARNED outcome-verifier held-out AUROC: {a['heldout_auroc_learned_outcome_verifier']}")
    print(f"   fraction_certified baseline AUROC:       {a['heldout_auroc_fraction_certified_baseline']}")
    print(f"   single-feature AUROCs: {a['single_feature_auroc_full']}")
    print(f"   gate (learned >= 0.80): {'PASS' if a['gate_pass'] else 'FAIL'}")
