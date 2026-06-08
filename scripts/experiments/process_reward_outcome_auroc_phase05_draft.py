"""DRAFT (Phase 0.5): does the verifier's per-step PROCESS-REWARD predict OUTCOME?

The step-level process-reward path is worth building ONLY if the verifier's aggregated
per-step reward RANKS correct traces above incorrect ones. Phase 0 showed trace-level
CERTIFICATION (all-steps-clean) is only 56% precise -- but a dense soft reward might
still rank traces usefully even if the hard pass/fail doesn't. This measures the
trace-OUTCOME AUROC of the aggregate process-reward, on existing p01 traces + the
existing ensemble. No training, no new infra.

GATE: best-aggregation trace-outcome AUROC >= 0.65 -> dense reward carries outcome
signal -> process-reward training is worth the harness build. ~0.5-0.55 -> the per-step
signal does not rank correct traces above incorrect at the trace level -> would
reward-hack; need an outcome-aware/stronger verifier first.

  .venv/bin/python scripts/experiments/process_reward_outcome_auroc_phase05_draft.py
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
OUT = REPO_ROOT / "results" / "process_reward_outcome_auroc_phase05.json"
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _chunks(text: str) -> list[str]:
    body = _THINK.sub("", text).strip()
    return [s.strip() for s in re.split(r"\n\s*\n", body)
            if len(s.strip()) >= 12 and re.search(r"[a-zA-Z0-9]", s)]


def _load(path: Path, limit: int | None = None) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ic = r.get("is_correct")
            ch = _chunks(str(r.get("text") or ""))
            if ic is None or not ch:
                continue
            out.append({"chunks": ch, "is_correct": bool(ic)})
            if limit and len(out) >= limit:
                break
    return out


def _auroc(labels: list[int], scores: list[float]) -> float | None:
    pos = [s for y, s in zip(labels, scores) if y == 1]
    neg = [s for y, s in zip(labels, scores) if y == 0]
    if not pos or not neg:
        return None
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def run(limit: int | None = None, write: bool = True) -> dict:
    traces = _load(TRACES, limit=limit)
    n = len(traces)
    gold = [1 if t["is_correct"] else 0 for t in traces]
    base_rate = sum(gold) / n if n else 0.0

    chunk_texts, owner = [], []
    for ti, t in enumerate(traces):
        for c in t["chunks"]:
            chunk_texts.append(c)
            owner.append(ti)
    panel = FoVerPanel(
        rows=tuple({"idx": i} for i in range(len(chunk_texts))),
        labels=tuple(0 for _ in chunk_texts),
        texts=tuple(chunk_texts),
        panel_sha256=hashlib.sha256("".join(chunk_texts).encode("utf-8")).hexdigest(),
    )
    scoring = score_carnot_ensemble(panel, REPO_ROOT)
    # per-step process reward = 1 - error_score (higher = more correct)
    reward = [1.0 - float(s) for s in scoring.scores]
    pred_correct = [1 - int(p) for p in scoring.error_preds]  # 1 if verifier says correct

    # Aggregate per-trace.
    per_trace_rewards: list[list[float]] = [[] for _ in range(n)]
    per_trace_predcorrect: list[list[int]] = [[] for _ in range(n)]
    for i, ti in enumerate(owner):
        per_trace_rewards[ti].append(reward[i])
        per_trace_predcorrect[ti].append(pred_correct[i])

    agg_mean = [sum(r) / len(r) for r in per_trace_rewards]
    agg_min = [min(r) for r in per_trace_rewards]
    agg_fraccert = [sum(p) / len(p) for p in per_trace_predcorrect]

    aurocs = {
        "mean_reward": _auroc(gold, agg_mean),
        "min_reward": _auroc(gold, agg_min),
        "fraction_certified": _auroc(gold, agg_fraccert),
    }
    best_name = max((k for k in aurocs if aurocs[k] is not None),
                    key=lambda k: aurocs[k], default=None)
    best = aurocs.get(best_name) if best_name else None
    gate_pass = bool(best is not None and best >= 0.65)
    verdict = (
        f"complete: process_reward_outcome_{'SIGNAL' if gate_pass else 'WEAK'}"
        f"_bestauroc{None if best is None else round(best,3)}_via_{best_name}"
        f"_baserate{base_rate:.3f}"
    )
    artifact = {
        "experiment": "process_reward_outcome_auroc_phase05_draft",
        "title": "process_reward_predicts_outcome",
        "honest_verdict": verdict,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "n_traces": n,
        "n_steps_scored": len(chunk_texts),
        "gold_correct_base_rate": round(base_rate, 4),
        "trace_outcome_auroc_by_aggregation": {k: (None if v is None else round(v, 4))
                                               for k, v in aurocs.items()},
        "best_aggregation": best_name,
        "best_trace_outcome_auroc": None if best is None else round(best, 4),
        "gate": "best trace-outcome AUROC >= 0.65 (dense process-reward carries outcome signal)",
        "gate_pass": gate_pass,
        "interpretation": (
            "Reference: a perfect OUTCOME verifier -> AUROC 1.0; chance -> 0.5. The moat's "
            "PER-STEP (in-format) AUROC was 0.967. This trace-OUTCOME AUROC of aggregated "
            "process-reward on FREE-FORM generations is the gate for the process-reward path. "
            "If it is ~0.5-0.6 the per-step signal does not transfer to outcome ranking -> "
            "process-reward training would reward-hack."
        ),
    }
    if write:
        OUT.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", "utf-8")
    return artifact


if __name__ == "__main__":
    art = run()
    print(f"-> {art['honest_verdict']}")
    print(f"   n_traces={art['n_traces']} steps={art['n_steps_scored']} base_rate={art['gold_correct_base_rate']}")
    print(f"   trace-outcome AUROC by aggregation: {art['trace_outcome_auroc_by_aggregation']}")
    print(f"   best: {art['best_aggregation']} = {art['best_trace_outcome_auroc']}")
    print(f"   Gate (>=0.65): {'PASS' if art['gate_pass'] else 'FAIL'}")
