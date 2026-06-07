"""DRAFT (#5, operator-facing): meta-verifier over our OWN experiment history.

THE QUESTION
------------
~688 milestone-entries of research, and the Depth-Over-Breadth rule exists
because most of it was breadth churn while the signal came from a handful of
decisive experiments (energy-descent negative, the verifier moat/efficiency
panel).  We have the Meta-EBM cascade-router theory and a conductor, but we have
NEVER pointed a discriminator at the question "which experiments are worth
running?"  This is the verify<<generate asymmetry applied recursively to our own
process: can a CHEAP pre-run signal predict that an experiment will be WASTED
(blocked / skipped / doomed / fabrication-flagged) or CHURN (vN+1 re-measurement
that moves no claim), before we burn the wall-clock?

This is a first FEASIBILITY build: it joins the task list in research-complete.yaml
to the honest_verdict in each results/experiment_*.json artifact, labels each task,
and asks whether pre-run features (title-scope keywords, has-exp-number, recency)
separate the WASTED tasks from the rest on a TEMPORAL holdout (train on early
milestones, test on recent).  If even a bag-of-title-keywords logistic model
beats chance, the planner could have flagged doomed/churn work at design time --
which is exactly what the Failed-Experiment-Rerun + Exclusion-Manifest rules try
to do by hand.

HONEST LIMITATIONS (stated up front, this is a draft):
- "decisive vs churn" has no ground-truth field; we use a verdict/title PROXY.
  WASTED is cleanly labelable (verdict tokens); CHURN is a title-keyword proxy.
- The join is by exp-number; pre-exp-number legacy tasks (p1-mX...) are unjoined
  and counted separately, not silently dropped.
- This MEASURES separability; it does NOT (yet) wire anything into the conductor.

CPU-only, read-only, observe-only on the conductor.  Run:
  .venv/bin/python scripts/experiments/meta_verifier_experiment_selection_draft.py
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_COMPLETE = REPO_ROOT / "research-complete.yaml"
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "experiment_meta_verifier_selection_draft.json"

# Verdict tokens that mark a task as WASTED wall-clock (cleanly labelable).
WASTED_TOKENS = (
    "blocked_",
    "doomed",
    "gate_block",
    "skipped",
    "not_viable",
    "still_",
)
# Title-scope buckets knowable BEFORE running (the cheap pre-run features).
SCOPE_KEYWORDS = {
    "churn_version": (r"\bv\d+\b", "rerun", "re-?audit", "reaudit", "re-?run"),
    "churn_aggregate": ("capstone", "archive", "activate", "matrix", "cross[-_]corpus",
                        "telemetry", "receipt", "sweep", "panel", "consolidat"),
    "hardware": ("kv260", "kria", "gatemate", "polarfire", "fpga", "bitstream", "ising tile"),
    "science_probe": ("probe", "scissor", "headroom", "ablation", "null[-_ ]space",
                     "diversity", "complementar", "independence", "fundamental"),
    "infra_harness": ("harness", "green[-_ ]gate", "schema", "reconcile", "lint",
                     "precondition", "unit[-_ ]test", "fixture"),
    "build_train": ("train", "distill", "reward", "grpo", "fr-11", "self-learning"),
}
# CHURN proxy: aggregate/version-rerun scope that is, by construction, re-measurement.
CHURN_BUCKETS = ("churn_version", "churn_aggregate")


def _verdict_of_artifact(path: Path) -> str | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    v = d.get("honest_verdict") or d.get("status")
    flagged = bool(d.get("flagged_adversarial"))
    base = str(v).strip().lower() if v is not None else ""
    return f"{base}\t{'FLAGGED' if flagged else ''}"


def _build_verdict_index() -> dict[int, str]:
    """Map exp-number -> verdict string from results/experiment_<n>_*.json."""

    index: dict[int, str] = {}
    for path in RESULTS_DIR.glob("experiment_*.json"):
        m = re.match(r"experiment_(\d+)", path.name)
        if not m:
            continue
        n = int(m.group(1))
        verdict = _verdict_of_artifact(path)
        if verdict is not None:
            # Keep the longest/most-specific verdict if multiple artifacts share a number.
            if n not in index or len(verdict) > len(index[n]):
                index[n] = verdict
    return index


def _exp_number(task_id: str) -> int | None:
    m = re.search(r"exp[-_]?(\d{3,4})", task_id.lower())
    return int(m.group(1)) if m else None


def _scope_features(title: str) -> dict[str, int]:
    low = title.lower()
    feats: dict[str, int] = {}
    for bucket, patterns in SCOPE_KEYWORDS.items():
        hit = any(re.search(p, low) for p in patterns)
        feats[bucket] = 1 if hit else 0
    return feats


def _label(verdict: str | None) -> str:
    """WASTED (clean) / COMPLETED, from the joined verdict string."""

    if verdict is None:
        return "UNJOINED"
    base, flag = (verdict.split("\t") + [""])[:2]
    if flag == "FLAGGED":
        return "WASTED"
    if any(tok in base for tok in WASTED_TOKENS):
        return "WASTED"
    if base.startswith(("complete", "success", "passed", "shipped")):
        return "COMPLETED"
    return "COMPLETED" if base else "UNJOINED"


def _auroc(labels: list[int], scores: list[float]) -> float | None:
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _fit_logreg(
    x: list[list[float]], y: list[int], *, iters: int = 400, lr: float = 0.3
) -> list[float]:
    """Tiny batch logistic regression (numpy-free) -> weight vector incl. bias."""

    n_feat = len(x[0])
    w = [0.0] * (n_feat + 1)  # last is bias
    n = len(x)
    for _ in range(iters):
        grad = [0.0] * (n_feat + 1)
        for xi, yi in zip(x, y):
            z = w[-1] + sum(w[j] * xi[j] for j in range(n_feat))
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            err = p - yi
            for j in range(n_feat):
                grad[j] += err * xi[j]
            grad[-1] += err
        for j in range(n_feat + 1):
            w[j] -= lr * grad[j] / n
    return w


def _predict(w: list[float], xi: list[float]) -> float:
    z = w[-1] + sum(w[j] * xi[j] for j in range(len(xi)))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def run(write: bool = True) -> dict[str, Any]:
    verdict_index = _build_verdict_index()
    milestones = yaml.safe_load(RESEARCH_COMPLETE.read_text(encoding="utf-8"))["milestones"]

    rows: list[dict[str, Any]] = []
    for order, entry in enumerate(milestones):
        tasks = entry.get("tasks")
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, dict):
                continue
            tid = str(task.get("id", ""))
            title = str(task.get("title", ""))
            n = _exp_number(tid)
            verdict = verdict_index.get(n) if n is not None else None
            label = _label(verdict)
            feats = _scope_features(title)
            rows.append(
                {
                    "milestone_order": order,
                    "task_id": tid,
                    "exp_number": n,
                    "label": label,
                    "is_wasted": 1 if label == "WASTED" else 0,
                    "is_churn_scope": 1 if any(feats[b] for b in CHURN_BUCKETS) else 0,
                    **feats,
                }
            )

    joined = [r for r in rows if r["label"] != "UNJOINED"]
    n_total = len(rows)
    n_joined = len(joined)
    n_wasted = sum(r["is_wasted"] for r in joined)
    n_churn_scope = sum(r["is_churn_scope"] for r in rows)

    # Temporal holdout: train on the first 70% of milestones, test on the last 30%.
    feat_keys = list(SCOPE_KEYWORDS.keys()) + ["is_churn_scope"]
    orders = sorted({r["milestone_order"] for r in joined})
    split = orders[int(0.7 * len(orders))] if orders else 0
    train = [r for r in joined if r["milestone_order"] < split]
    test = [r for r in joined if r["milestone_order"] >= split]

    auroc = None
    weights = None
    if train and test and sum(r["is_wasted"] for r in train) > 0:
        xtr = [[float(r[k]) for k in feat_keys] for r in train]
        ytr = [r["is_wasted"] for r in train]
        w = _fit_logreg(xtr, ytr)
        weights = dict(zip([*feat_keys, "bias"], [round(v, 4) for v in w]))
        xte = [[float(r[k]) for k in feat_keys] for r in test]
        yte = [r["is_wasted"] for r in test]
        scores = [_predict(w, xi) for xi in xte]
        auroc = _auroc(yte, scores)

    # Per-bucket wasted-rate lift (which scope classes waste the most wall-clock).
    bucket_lift = {}
    base_rate = n_wasted / n_joined if n_joined else 0.0
    for bucket in SCOPE_KEYWORDS:
        inb = [r for r in joined if r[bucket] == 1]
        if inb:
            rate = sum(r["is_wasted"] for r in inb) / len(inb)
            bucket_lift[bucket] = {
                "n": len(inb),
                "wasted_rate": round(rate, 4),
                "lift_vs_base": round(rate / base_rate, 2) if base_rate else None,
            }

    artifact = {
        "experiment": "meta_verifier_selection_draft",
        "title": "meta_verifier_experiment_selection",
        "honest_verdict": (
            f"complete: meta_verifier_feasibility_wasted_auroc{auroc:.3f}_base{base_rate:.3f}"
            if auroc is not None
            else "complete: meta_verifier_feasibility_insufficient_holdout_signal"
        ),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "n_tasks_total": n_total,
        "n_tasks_joined_to_verdict": n_joined,
        "n_unjoined_legacy_or_missing": n_total - n_joined,
        "base_wasted_rate": round(base_rate, 4),
        "n_wasted": n_wasted,
        "n_churn_scope_titles": n_churn_scope,
        "churn_scope_fraction": round(n_churn_scope / n_total, 4) if n_total else 0.0,
        "holdout_wasted_auroc": auroc,
        "holdout_split_milestone_order": split,
        "logreg_weights": weights,
        "per_scope_wasted_lift": bucket_lift,
        "label_proxy_caveat": (
            "WASTED = verdict blocked_/doomed/skipped/flagged (clean). CHURN = title-scope "
            "proxy (no ground-truth decisiveness field). Join is by exp-number; legacy "
            "p1-* tasks are counted as unjoined, not dropped."
        ),
    }
    if write:
        OUTPUT_PATH.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")
    return artifact


if __name__ == "__main__":
    art = run()
    print(f"meta-verifier feasibility -> {art['honest_verdict']}")
    print(
        f"  tasks: {art['n_tasks_joined_to_verdict']}/{art['n_tasks_total']} joined | "
        f"wasted base-rate {art['base_wasted_rate']} | churn-scope frac {art['churn_scope_fraction']}"
    )
    print(f"  holdout WASTED-prediction AUROC: {art['holdout_wasted_auroc']}")
    print("  per-scope wasted lift:")
    for b, v in sorted(art["per_scope_wasted_lift"].items(), key=lambda kv: -(kv[1]["lift_vs_base"] or 0)):
        print(f"    {b:18s} n={v['n']:5d}  wasted_rate={v['wasted_rate']:.3f}  lift={v['lift_vs_base']}")
