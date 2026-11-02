"""DRAFT (#2 verifier domain expansion): can a CHEAP verifier approximate the FORMAL
ORACLE on CODE correctness? — the stage-2 template for minting a new-domain verifier.

WHY this is the ARC-AGI-3 path (north-star.md §0 stage 2). A new domain is cheap to
verify-train when a FORMAL ORACLE gives gold labels for free. For code that oracle is
TEST EXECUTION (already run -> data/code_verification_corpus_v{1,2}.jsonl carry
test_outcome + label). The verifier's job is to be a CHEAP proxy for that expensive
oracle (don't execute untrusted code per candidate at inference). This experiment asks:
how well do cheap, NON-EXECUTING features predict the oracle's pass/fail label?

  v2: 60 HumanEval, BALANCED 30/30, failures are SEMANTIC (return None stub, wrong
      logic -- ALL parse fine) -> the HARD, honest test.
  v1: 320 MBPP/HumanEval, 296 fail/24 pass, mostly SYNTAX/EXTRACTION failures -> easy
      sanity baseline (ast.parse alone separates most).

Features are all CHEAP and SAFE -- ast.parse only PARSES, never EXECUTES; the Carnot
math ensemble is a TRANSFER probe (does the math-step verifier carry any code signal?).
No candidate code is ever run here; labels come from the pre-computed oracle.

Gate: on the BALANCED v2 corpus, learned held-out outcome-AUROC >= 0.70 means a cheap
code outcome-verifier signal exists (mint a `code` registry entry). ~0.5 means cheap
static verification has a CEILING on semantic code correctness -> the formal oracle
(execution) is load-bearing -> code domain is execution-bound (the honest finding,
mirrors math being self-consistency-bound).

  .venv/bin/python scripts/experiments/code_verifier_formal_oracle_derisk_draft.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from outcome_verifier_learn_derisk_draft import _auroc, _fit_logreg, _predict  # noqa: E402

from carnot.eval.verifier_error_independence_scissor_at_scale import (  # noqa: E402
    FoVerPanel, score_carnot_ensemble,
)
from carnot.verify.ast_structure_verifier import ASTStructureVerifier  # noqa: E402

CORPORA = {
    "v2_humaneval_balanced": REPO_ROOT / "data" / "code_verification_corpus_v2.jsonl",
    "v1_mbpp_humaneval_mixed": REPO_ROOT / "data" / "code_verification_corpus_v1.jsonl",
}
OUT = REPO_ROOT / "results" / "code_verifier_formal_oracle_derisk.json"
# ast_struct_verifier_score = the SHIPPED dedicated code verifier (structural violation
# energy). On v2 (semantic, all parse) it should also be ~chance -> closes the "did a
# DEDICATED static code verifier help?" caveat with a real shipped verifier, not a proxy.
FNAMES = ["ast_parse_ok", "n_lines", "char_len_k", "returns_none_stub", "has_loop",
          "has_cond", "n_defs", "carnot_math_transfer_reward", "ast_struct_verifier_score"]
_ASTV = ASTStructureVerifier()


def _ast_ok(code):
    try:
        ast.parse(code)  # PARSE ONLY -- never executes the candidate.
        return True
    except Exception:
        return False


def _returns_none_stub(code):
    """Heuristic: a body whose only non-doc statement is `return None`/`return`/`pass`."""
    try:
        tree = ast.parse(code)
    except Exception:
        return 0.0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            body = [s for s in node.body if not (isinstance(s, ast.Expr)
                    and isinstance(getattr(s, "value", None), ast.Constant))]
            if len(body) == 1 and isinstance(body[0], (ast.Return, ast.Pass)):
                v = getattr(body[0], "value", None)
                if v is None or (isinstance(v, ast.Constant) and v.value is None):
                    return 1.0
    return 0.0


def _struct_feats(code):
    has_loop = 1.0 if any(k in code for k in ("for ", "while ")) else 0.0
    has_cond = 1.0 if ("if " in code) else 0.0
    n_defs = float(code.count("def "))
    return has_loop, has_cond, n_defs


def _carnot_transfer(codes):
    """Carnot MATH-step ensemble scored on code-as-text: a pure TRANSFER probe."""
    chunk_texts, owner = [], []
    for ci, code in enumerate(codes):
        for line in [s.strip() for s in str(code).splitlines() if len(s.strip()) >= 8]:
            chunk_texts.append(line)
            owner.append(ci)
    if not chunk_texts:
        return [0.0] * len(codes)
    panel = FoVerPanel(rows=tuple({"idx": i} for i in range(len(chunk_texts))),
                       labels=tuple(0 for _ in chunk_texts), texts=tuple(chunk_texts),
                       panel_sha256=hashlib.sha256("".join(chunk_texts).encode()).hexdigest())
    scoring = score_carnot_ensemble(panel, REPO_ROOT)
    pcorrect = [1 - int(p) for p in scoring.error_preds]
    agg = {}
    for i, ci in enumerate(owner):
        agg.setdefault(ci, []).append(pcorrect[i])
    return [(sum(agg.get(ci, [0])) / len(agg.get(ci, [0]))) if agg.get(ci) else 0.0
            for ci in range(len(codes))]


def _features(rows):
    codes = [str(r.get("candidate_code") or "") for r in rows]
    transfer = _carnot_transfer(codes)
    feats, labels = [], []
    for code, tr, r in zip(codes, transfer, rows):
        hl, hc, nd = _struct_feats(code)
        # dedicated shipped verifier returns VIOLATION energy (higher=worse); use as-is
        # (the logistic learns sign). For a correctness-AUROC it is 1 - energy direction.
        astv = float(_ASTV.score(code))
        feats.append([
            1.0 if _ast_ok(code) else 0.0,
            float(len(code.splitlines())),
            float(len(code)) / 1000.0,
            _returns_none_stub(code),
            hl, hc, nd, float(tr), astv,
        ])
        labels.append(int(bool(r.get("label"))))
    return feats, labels


def _eval_corpus(name, path, seed=0):
    import random
    rows = [json.loads(l) for l in path.open()]
    feats, labels = _features(rows)
    n = len(feats)
    base = sum(labels) / n
    single = {FNAMES[j]: round(_auroc(labels, [f[j] for f in feats]) or 0.5, 4)
              for j in range(len(FNAMES))}
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    cut = int(0.7 * n)
    tr, te = idx[:cut], idx[cut:]
    if min(sum(labels[i] for i in te), len(te) - sum(labels[i] for i in te)) < 1:
        learned = None  # held-out has only one class -> AUROC undefined
    else:
        w, m, s = _fit_logreg([feats[i] for i in tr], [labels[i] for i in tr])
        learned = _auroc([labels[i] for i in te], [_predict(w, m, s, feats[i]) for i in te])
    return {
        "n": n, "pass_base_rate": round(base, 4), "n_heldout": len(te),
        "single_feature_auroc": single,
        "learned_heldout_auroc": None if learned is None else round(learned, 4),
        "best_single_feature": max(single, key=single.get),
        "best_single_auroc": round(max(single.values()), 4),
    }


def run(write=True):
    results = {name: _eval_corpus(name, path) for name, path in CORPORA.items() if path.is_file()}
    v2 = results.get("v2_humaneval_balanced", {})
    v2_auroc = v2.get("learned_heldout_auroc")
    gate = bool(v2_auroc is not None and v2_auroc >= 0.70)
    if v2_auroc is None:
        finding = "v2_undefined_heldout"
    elif gate:
        finding = "CHEAP_CODE_VERIFIER_SIGNAL_EXISTS"
    elif v2_auroc >= 0.60:
        finding = "weak_signal_below_gate"
    else:
        finding = "EXECUTION_BOUND_cheap_static_at_ceiling"
    verdict = (f"complete: code_formal_oracle_derisk_{finding}"
               f"_v2learned{v2_auroc}_v2best{v2.get('best_single_feature')}"
               f"{v2.get('best_single_auroc')}")
    art = {
        "experiment": "code_verifier_formal_oracle_derisk_draft",
        "title": "code_domain_verifier_vs_formal_oracle",
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "label_source": "formal_oracle_test_execution_precomputed",
        "per_corpus": results,
        "gate": "v2 (balanced semantic) learned held-out outcome-AUROC >= 0.70",
        "gate_pass": gate,
        "interpretation": (
            "v2 is the HARD test (balanced, semantic failures all parse). If learned >= 0.70 a "
            "cheap code outcome-verifier exists -> mint a `code` registry entry + add to ARC "
            "domain set. If ~0.5 -> static verification is EXECUTION-BOUND on semantic code "
            "(the formal oracle is load-bearing), mirroring math being self-consistency-bound: "
            "the honest domain-coverage map the harness routes on. v1 (syntax-dominated) is the "
            "sanity baseline -> ast_parse_ok should dominate there."
        ),
        "no_code_executed": True,
        "principle_label_source": "formal oracle (test execution) gives free gold labels -- the "
                                  "stage-2 mechanism for minting a verifier in a new domain",
    }
    if write:
        OUT.write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    return art


if __name__ == "__main__":
    a = run()
    print(f"-> {a['honest_verdict']}")
    for name, r in a["per_corpus"].items():
        print(f"   [{name}] n={r['n']} base={r['pass_base_rate']} "
              f"learned_heldout_auroc={r['learned_heldout_auroc']} "
              f"best_single={r['best_single_feature']}({r['best_single_auroc']})")
    print(f"   gate (v2 learned >= 0.70): {'PASS' if a['gate_pass'] else 'FAIL'}")
