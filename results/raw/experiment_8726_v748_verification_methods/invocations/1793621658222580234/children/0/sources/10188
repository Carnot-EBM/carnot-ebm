"""DRAFT (Phase 0 v3 / Path 1): certification precision IN the verifier's OWN format.

v2 found low certification precision on free-form p01 steps -- but the verifier is
calibrated on FoVer (problem,step)-chunk format at a specific granularity, so v2
conflated "AUROC->precision" with "format transfer". v3 isolates the first question:
measure CERTIFICATION PRECISION on the FoVer corpus ITSELF (in-distribution,
in-format, the exact step_text chunks the 0.91-AUROC ensemble was measured on).

  - If precision is HIGH here (>=85% at usable recall): the verifier CAN certify
    in-format; the p01 failure is a solvable format/chunking-transfer problem.
  - If precision is ALSO insufficient here: 0.91 AUROC inherently does not yield
    high certification precision -- a fundamental precision-recall limit that bounds
    verifier-as-RFT-reward regardless of format. The decisive disambiguation.

  .venv/bin/python scripts/experiments/verifier_certification_precision_phase0_v3_infomat_draft.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from carnot.eval.verifier_error_independence_scissor_at_scale import (
    FoVerPanel,
    score_carnot_ensemble,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FOVER = REPO_ROOT / "data" / "fover_corpus_v4.json"
OUT = REPO_ROOT / "results" / "verifier_certification_precision_phase0_v3_infomat.json"


def _load_fover(path: Path) -> tuple[list[str], list[int]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = d if isinstance(d, list) else (d.get("items") or d.get("rows") or [])
    texts, correct = [], []  # correct: 1 if the step is actually correct
    for r in rows:
        t = str(r.get("step_text") or "")
        lab = str(r.get("label") or "").strip().lower()
        if not t.strip() or lab not in {"correct", "incorrect"}:
            continue
        texts.append(t)
        correct.append(1 if lab == "correct" else 0)
    return texts, correct


def _balance(texts: list[str], correct: list[int], seed: int = 3917) -> tuple[list[str], list[int]]:
    """All incorrect + an equal seeded sample of correct (the moat's balanced slice).

    The raw FoVer corpus is ~98% correct, which makes certification precision trivially
    ~base-rate. The 0.91 AUROC was measured on a BALANCED slice; replicate that so
    precision is meaningful (50% base rate)."""

    import random

    inc = [i for i in range(len(correct)) if correct[i] == 0]
    cor = [i for i in range(len(correct)) if correct[i] == 1]
    rng = random.Random(seed)
    rng.shuffle(cor)
    keep = inc + cor[: len(inc)]
    rng.shuffle(keep)
    return [texts[i] for i in keep], [correct[i] for i in keep]


def run(write: bool = True) -> dict:
    texts, correct = _load_fover(FOVER)
    texts, correct = _balance(texts, correct)
    n = len(texts)
    base_rate = sum(correct) / n if n else 0.0

    panel = FoVerPanel(
        rows=tuple({"idx": i} for i in range(n)),
        labels=tuple(0 if c else 1 for c in correct),  # error label
        texts=tuple(texts),
        panel_sha256=hashlib.sha256("".join(texts).encode("utf-8")).hexdigest(),
    )
    scoring = score_carnot_ensemble(panel, REPO_ROOT)
    err = [float(s) for s in scoring.scores]  # higher = more error

    # Default operating point (ensemble's own threshold).
    cert_def = [i for i in range(n) if int(scoring.error_preds[i]) == 0]
    def_prec = sum(correct[i] for i in cert_def) / len(cert_def) if cert_def else None
    def_rec = sum(correct[i] for i in cert_def) / (sum(correct) or 1) if cert_def else None

    # Sweep: certify-correct iff error-score <= t.
    lo, hi = min(err), max(err)
    grid = [lo + (hi - lo) * k / 40 for k in range(41)]
    sweep = []
    for t in grid:
        cert = [i for i in range(n) if err[i] <= t]
        if not cert:
            sweep.append({"threshold": round(t, 4), "n_certified": 0,
                          "precision": None, "recall_of_correct": None})
            continue
        tp = sum(correct[i] for i in cert)
        sweep.append({"threshold": round(t, 4), "n_certified": len(cert),
                      "precision": round(tp / len(cert), 4),
                      "recall_of_correct": round(tp / (sum(correct) or 1), 4)})
    usable = [p for p in sweep if p["recall_of_correct"] and p["recall_of_correct"] >= 0.20]
    rft_point = max(usable, key=lambda p: p["precision"]) if usable else None
    # also: highest recall at which precision still >= 0.85
    hi_prec = [p for p in sweep if p["precision"] and p["precision"] >= 0.85]
    recall_at_85 = max((p["recall_of_correct"] for p in hi_prec), default=0.0)

    gate_pass = bool(rft_point and rft_point["precision"] and rft_point["precision"] >= 0.85)
    verdict = (
        f"complete: infomat_certification_{'VIABLE' if gate_pass else 'INSUFFICIENT'}"
        f"_baserate{base_rate:.3f}_rftprec{rft_point['precision'] if rft_point else 'na'}"
        f"_recallAt85prec{round(recall_at_85,3)}"
    )
    artifact = {
        "experiment": "verifier_certification_precision_phase0_v3_infomat_draft",
        "title": "verifier_certification_precision_in_fover_format",
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "corpus": "fover_corpus_v4 (in-distribution, in-format)",
        "n_steps": n,
        "correct_base_rate": round(base_rate, 4),
        "default_threshold_precision": None if def_prec is None else round(def_prec, 4),
        "default_threshold_recall": None if def_rec is None else round(def_rec, 4),
        "rft_operating_point": rft_point,
        "recall_at_85pct_precision": round(recall_at_85, 4),
        "precision_recall_sweep": sweep,
        "phase1_gate": "certification precision >= 0.85 at recall >= 0.20",
        "phase1_gate_pass": gate_pass,
        "interpretation": (
            "If VIABLE: the verifier certifies precisely IN-FORMAT; p01's failure is a "
            "solvable format/chunking transfer. If INSUFFICIENT: 0.91 AUROC does NOT yield "
            "high certification precision even in-distribution -> a fundamental bound on "
            "verifier-as-RFT-reward (and a precision-recall caution about citing AUROC)."
        ),
    }
    if write:
        OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")
    return artifact


if __name__ == "__main__":
    art = run()
    print(f"-> {art['honest_verdict']}")
    print(f"   n_steps={art['n_steps']} correct_base_rate={art['correct_base_rate']}")
    print(f"   default-threshold precision={art['default_threshold_precision']} recall={art['default_threshold_recall']}")
    print(f"   RFT operating point (best precision @ recall>=0.20): {art['rft_operating_point']}")
    print(f"   recall achievable at >=85% precision: {art['recall_at_85pct_precision']}")
    print(f"   Phase-1 gate: {'PASS' if art['phase1_gate_pass'] else 'FAIL'}")
