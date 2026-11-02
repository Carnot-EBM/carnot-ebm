"""DRAFT (Phase 0 v2): PER-STEP verifier certification precision.

v1 fed whole traces to a per-STEP process verifier -> base-rate (format mismatch).
v2 uses the verifier the way it works: score each reasoning STEP, then certify a
TRACE as correct iff none of its steps is flagged as an error (the natural way RFT
would filter -- keep a generated trace only if every step checks out). Measures the
certification PRECISION that decides whether an imperfect (0.91-AUROC) verifier can
drive self-improvement without poisoning the RFT training set.

Two aggregations:
  (A) ALL-CLEAN rule: certify iff no step has error_pred==1 (the ensemble's own thr).
  (B) SCORE sweep: trace_error = max over steps of step error-score; certify iff
      trace_error <= t; sweep t for the precision-recall curve and the RFT point.

  .venv/bin/python scripts/experiments/verifier_certification_precision_phase0_v2_draft.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from carnot.eval.verifier_error_independence_scissor_at_scale import (
    FoVerPanel,
    score_carnot_ensemble,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACES = REPO_ROOT / "data" / "p01_difficulty_matched_generations_flattened_v2.jsonl"
OUT = REPO_ROOT / "results" / "verifier_certification_precision_phase0_v2.json"

_MARKUP = {"<think>", "</think>", "", "<answer>", "</answer>"}


def _is_substantive(step: str) -> bool:
    s = step.strip()
    return s not in _MARKUP and len(s) >= 8 and bool(re.search(r"[a-zA-Z0-9]", s))


def _load(path: Path, limit: int | None = None) -> list[dict]:
    traces = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            steps = [s for s in (r.get("reasoning_steps") or []) if _is_substantive(str(s))]
            ic = r.get("is_correct")
            if ic is None or not steps:
                continue
            traces.append({"steps": [str(s) for s in steps], "is_correct": bool(ic)})
            if limit and len(traces) >= limit:
                break
    return traces


def run(limit: int | None = None, write: bool = True) -> dict:
    traces = _load(TRACES, limit=limit)
    # Flatten every step into one panel; remember which trace each step belongs to.
    step_texts: list[str] = []
    owner: list[int] = []
    for ti, t in enumerate(traces):
        for s in t["steps"]:
            step_texts.append(s)
            owner.append(ti)

    panel = FoVerPanel(
        rows=tuple({"idx": i} for i in range(len(step_texts))),
        labels=tuple(0 for _ in step_texts),  # unused for scoring
        texts=tuple(step_texts),
        panel_sha256=hashlib.sha256("".join(step_texts).encode("utf-8")).hexdigest(),
    )
    scoring = score_carnot_ensemble(panel, REPO_ROOT)
    step_err_score = [float(s) for s in scoring.scores]
    step_err_pred = [int(p) for p in scoring.error_preds]

    n_tr = len(traces)
    gold = [1 if t["is_correct"] else 0 for t in traces]
    base_rate = sum(gold) / n_tr if n_tr else 0.0

    # Per-trace: max step error score, and any-flagged.
    trace_max_err = [0.0] * n_tr
    trace_any_flag = [0] * n_tr
    for i, ti in enumerate(owner):
        trace_max_err[ti] = max(trace_max_err[ti], step_err_score[i])
        if step_err_pred[i] == 1:
            trace_any_flag[ti] = 1

    # (A) ALL-CLEAN rule.
    clean = [i for i in range(n_tr) if trace_any_flag[i] == 0]
    a_prec = (sum(gold[i] for i in clean) / len(clean)) if clean else None
    a_rec = (sum(gold[i] for i in clean) / (sum(gold) or 1)) if clean else None

    # (B) SCORE sweep on trace_max_err (certify iff max-step-error <= t).
    lo, hi = min(trace_max_err), max(trace_max_err)
    grid = [lo + (hi - lo) * k / 20 for k in range(21)]
    sweep = []
    for t in grid:
        cert = [i for i in range(n_tr) if trace_max_err[i] <= t]
        if not cert:
            sweep.append({"threshold": round(t, 4), "n_certified": 0, "precision": None,
                          "recall_of_correct": None})
            continue
        tp = sum(gold[i] for i in cert)
        sweep.append({
            "threshold": round(t, 4), "n_certified": len(cert),
            "precision": round(tp / len(cert), 4),
            "recall_of_correct": round(tp / (sum(gold) or 1), 4),
        })
    usable = [p for p in sweep if p["recall_of_correct"] and p["recall_of_correct"] >= 0.20]
    rft_point = max(usable, key=lambda p: p["precision"]) if usable else None
    best_prec = max((p["precision"] for p in sweep if p["precision"] is not None), default=None)

    gate_pass = bool(rft_point and rft_point["precision"] and rft_point["precision"] >= 0.85)
    verdict = (
        f"complete: perstep_certification_{'VIABLE' if gate_pass else 'INSUFFICIENT'}"
        f"_baserate{base_rate:.3f}_allcleanprec{a_prec if a_prec is None else round(a_prec,3)}"
        f"_bestprec{best_prec}_rftprec{rft_point['precision'] if rft_point else 'na'}"
    )
    artifact = {
        "experiment": "verifier_certification_precision_phase0_v2_draft",
        "title": "verifier_certification_precision_perstep",
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "n_traces": n_tr,
        "n_steps_scored": len(step_texts),
        "gold_correct_base_rate": round(base_rate, 4),
        "all_clean_rule": {"n_certified": len(clean),
                           "precision": None if a_prec is None else round(a_prec, 4),
                           "recall_of_correct": None if a_rec is None else round(a_rec, 4)},
        "best_precision_any_threshold": best_prec,
        "rft_operating_point": rft_point,
        "precision_recall_sweep": sweep,
        "phase1_gate": "per-step certification precision >= 0.85 at recall >= 0.20",
        "phase1_gate_pass": gate_pass,
        "trace_source": str(TRACES.relative_to(REPO_ROOT)),
        "caveat": (
            "DRAFT. Per-step certification via the production text-verifier ensemble (each "
            "reasoning step scored, trace certified iff all steps clean / max-step-error <= t). "
            "If precision stays ~base-rate, the text verifiers do not discriminate correct from "
            "incorrect REASONING steps in this corpus (they target arithmetic/logic surface "
            "errors) -> bounds verifier-as-RFT-reward. Follow-up: the FoVer (problem,step)-format "
            "model verifiers, harder corpus (gsm8k/hardmath)."
        ),
    }
    if write:
        OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")
    return artifact


if __name__ == "__main__":
    art = run()
    print(f"-> {art['honest_verdict']}")
    print(f"   n_traces={art['n_traces']} steps={art['n_steps_scored']} base_rate={art['gold_correct_base_rate']}")
    print(f"   ALL-CLEAN rule: {art['all_clean_rule']}")
    print(f"   best precision (any thr): {art['best_precision_any_threshold']}")
    print(f"   RFT operating point: {art['rft_operating_point']}")
    print(f"   Phase-1 gate: {'PASS' if art['phase1_gate_pass'] else 'FAIL'}")
