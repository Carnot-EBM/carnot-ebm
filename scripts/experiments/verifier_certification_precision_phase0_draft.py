"""DRAFT (Phase 0, operator-facing): can Carnot's IMPERFECT verifier CERTIFY?

The #4 v4 Sudoku result (verifier teaches a generator via RFT) used a PERFECT
verifier. Scaling to reasoning, Carnot's verifier ensemble is 0.91 AUROC -- imperfect.
RFT cares about CERTIFICATION PRECISION, not AUROC: of the traces the verifier
certifies as correct, what fraction actually are? High precision -> RFT trains on
mostly-correct data and self-improves; low precision -> false-positive traces poison
training and it collapses (the Sudoku soft-argmin failure, at scale).

This measures it directly on EXISTING data + the EXISTING ensemble -- no fine-tuning,
no generation. Reuses real p01 generated traces (with gold is_correct) and the
production score_carnot_ensemble (text verifiers, CPU). Gate Phase 1 (RFT) on the
result: precision >= ~85% at a recall that yields enough certified traces to train on.

  .venv/bin/python scripts/experiments/verifier_certification_precision_phase0_draft.py
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
TRACES = REPO_ROOT / "data" / "p01_difficulty_matched_generations_flattened_v2.jsonl"
OUT = REPO_ROOT / "results" / "verifier_certification_precision_phase0.json"


def _load_traces(path: Path, limit: int | None = None) -> tuple[list[str], list[int]]:
    texts: list[str] = []
    labels: list[int] = []  # error label: 1 = incorrect, 0 = correct (FoVer convention)
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = str(r.get("text") or "")
            ic = r.get("is_correct")
            if ic is None or not t.strip():
                continue
            texts.append(t)
            labels.append(0 if bool(ic) else 1)
            if limit and len(texts) >= limit:
                break
    return texts, labels


def _precision_recall_at(scores: list[float], gold_correct: list[int], thr: float) -> dict:
    """Certify-correct iff error-score <= thr. Precision/recall of the certified set."""

    certified = [i for i, s in enumerate(scores) if s <= thr]
    if not certified:
        return {"threshold": thr, "n_certified": 0, "precision": None, "recall": None}
    tp = sum(gold_correct[i] for i in certified)
    n_correct_total = sum(gold_correct) or 1
    return {
        "threshold": round(thr, 4),
        "n_certified": len(certified),
        "precision": round(tp / len(certified), 4),
        "recall_of_correct": round(tp / n_correct_total, 4),
    }


def run(limit: int | None = None, write: bool = True) -> dict:
    texts, labels = _load_traces(TRACES, limit=limit)
    n = len(texts)
    gold_correct = [1 - e for e in labels]  # 1 if the trace is actually correct
    base_rate = sum(gold_correct) / n if n else 0.0

    panel = FoVerPanel(
        rows=tuple({"idx": i} for i in range(n)),
        labels=tuple(labels),
        texts=tuple(texts),
        panel_sha256=hashlib.sha256("".join(texts).encode("utf-8")).hexdigest(),
    )
    scoring = score_carnot_ensemble(panel, REPO_ROOT)
    scores = [float(s) for s in scoring.scores]  # error scores: higher = more error

    # Default operating point (the ensemble's own threshold): certify = error_pred 0.
    certified_default = [i for i, p in enumerate(scoring.error_preds) if int(p) == 0]
    def_prec = (
        sum(gold_correct[i] for i in certified_default) / len(certified_default)
        if certified_default else None
    )
    def_recall = (
        sum(gold_correct[i] for i in certified_default) / (sum(gold_correct) or 1)
        if certified_default else None
    )

    # Sweep the error-score threshold (lower = stricter certification = higher precision).
    lo, hi = min(scores), max(scores)
    grid = [lo + (hi - lo) * k / 20 for k in range(21)]
    sweep = [_precision_recall_at(scores, gold_correct, t) for t in grid]

    # The RFT operating point: strictest threshold that still certifies >= 20% of correct.
    usable = [p for p in sweep if p["recall_of_correct"] and p["recall_of_correct"] >= 0.20]
    rft_point = min(usable, key=lambda p: p["threshold"]) if usable else None

    gate_pass = bool(rft_point and rft_point["precision"] and rft_point["precision"] >= 0.85)
    verdict = (
        f"complete: certification_{'VIABLE' if gate_pass else 'INSUFFICIENT'}"
        f"_baserate{base_rate:.3f}_defaultprec{def_prec if def_prec is None else round(def_prec,3)}"
        f"_rftprec{rft_point['precision'] if rft_point else 'na'}"
    )
    artifact = {
        "experiment": "verifier_certification_precision_phase0_draft",
        "title": "verifier_certification_precision",
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "n_traces": n,
        "gold_correct_base_rate": round(base_rate, 4),
        "default_threshold_precision": None if def_prec is None else round(def_prec, 4),
        "default_threshold_recall": None if def_recall is None else round(def_recall, 4),
        "rft_operating_point": rft_point,
        "precision_recall_sweep": sweep,
        "phase1_gate": "certification precision >= 0.85 at recall >= 0.20",
        "phase1_gate_pass": gate_pass,
        "trace_source": str(TRACES.relative_to(REPO_ROOT)),
        "caveat": (
            "DRAFT. Whole-trace certification via the production text-verifier ensemble. "
            "Precision is the load-bearing metric for RFT (poisons training if low). A "
            "step-level certification + a harder corpus (gsm8k/hardmath) are follow-ups. "
            "If precision is high only at very low recall, RFT has few traces to learn from."
        ),
    }
    if write:
        OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")
    return artifact


if __name__ == "__main__":
    art = run()
    print(f"-> {art['honest_verdict']}")
    print(f"   n={art['n_traces']} gold_correct_base_rate={art['gold_correct_base_rate']}")
    print(f"   default-threshold precision={art['default_threshold_precision']} "
          f"recall={art['default_threshold_recall']}")
    print(f"   RFT operating point: {art['rft_operating_point']}")
    print(f"   Phase-1 gate (prec>=0.85 @ recall>=0.20): {'PASS' if art['phase1_gate_pass'] else 'FAIL'}")
