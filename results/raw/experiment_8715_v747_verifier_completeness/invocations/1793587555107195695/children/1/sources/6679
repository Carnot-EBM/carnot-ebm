#!/usr/bin/env python3
"""Exp 1212 v2 — Tier 1 Online Constraint Addition (vs. exp134 reweighting baseline).

Context:
    Exp 134 showed precision reweighting of existing constraints is flat (0% improvement).
    research-program.md Tier 1 proposes constraint ADDITION: when the agent observes
    that constraint type X fires in >60% of wrong responses but <20% of correct ones,
    ADD a new specialised constraint Y derived from X's firing context.

    This experiment operationalises the ADDITION strategy on 100 FoVer corpus samples:
    - Learning set (rows 0–49): ConstraintAdditionAgent observes arithmetic vs. logic
      firing rates split by ground-truth label.
    - Holdout (rows 50–99): evaluate precision before addition (any constraint present)
      vs. after addition (only high-signal constraint types).

    The "before" mode represents a naïve pipeline that has no learning: it flags any
    response where ANY constraint is extracted (arithmetic, logic, NL), regardless of
    whether that type is discriminative.  This mimics a uniform-weight policy.

    The "after" mode is the learned pipeline: only constraint types detected as
    high-signal (wrong_rate > 0.6 AND correct_rate < 0.2) are used for classification,
    suppressing noise from non-discriminative types.

Spec: REQ-LEARN-1212, SCENARIO-LEARN-1212
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _maybe_reexec_repo_venv() -> None:
    """Re-exec under the repo .venv so the documented run command works."""
    if __name__ != "__main__":
        return
    venv_python = _REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return
    if str(venv_python.resolve()) == str(Path(sys.executable).resolve()):
        return
    import os

    os.execv(str(venv_python), [str(venv_python)] + sys.argv)


_maybe_reexec_repo_venv()

for _p in (str(_REPO_ROOT / "python"), str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RESULT_PATH = _REPO_ROOT / "results" / "experiment_1212_tier1_constraint_addition_v2.json"
FOVER_JSONL = _REPO_ROOT / "data" / "fover_corpus.jsonl"

# FoVer corpus split: first 50 for learning, next 50 for holdout
N_LEARN = 50
N_HOLDOUT = 50
FOVER_OFFSET = 0  # start from the beginning

# exp134 reweighting baseline — known from prior experiment
EXP134_REWEIGHTING_BASELINE = 0.0

# Thresholds for ConstraintAdditionAgent
WRONG_THRESHOLD = 0.6
CORRECT_THRESHOLD = 0.2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_artifact(payload: dict[str, Any]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, indent=2))


def _load_fover(n: int, offset: int = 0) -> list[dict]:
    rows = []
    with open(FOVER_JSONL) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if len(rows) >= n:
                break
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Constraint-firing helpers
# ---------------------------------------------------------------------------


def _get_violated_types(text: str, arith_ext: Any, logic_ext: Any, nl_ext: Any) -> set[str]:
    """Return set of constraint type tags that have at least one violation.

    Arithmetic violations are definitive (satisfied=False in metadata).
    Logic and NL constraints are extracted but not checksummed, so their
    *presence* is used as a proxy for "fires" (mimicking the naïve pipeline
    that treats any extracted constraint as a potential flag).
    """
    fired: set[str] = set()

    arith_results = arith_ext.extract(text)
    if any(not r.metadata.get("satisfied", True) for r in arith_results):
        fired.add("arithmetic")

    if logic_ext.extract(text):
        fired.add("logic")

    if nl_ext.extract(text):
        fired.add("nl")

    return fired


def _predict_before(text: str, arith_ext: Any, logic_ext: Any, nl_ext: Any) -> bool:
    """BEFORE mode: flag if any constraint type fires (arith violation OR logic OR NL present)."""
    fired = _get_violated_types(text, arith_ext, logic_ext, nl_ext)
    return bool(fired)


def _predict_after(text: str, high_signal_types: list[str], arith_ext: Any) -> bool:
    """AFTER mode: flag only if a high-signal constraint type fires.

    For arithmetic (the detected high-signal type on FoVer), fires on actual
    arithmetic violations only — not just presence of arithmetic patterns.
    """
    if "arithmetic" in high_signal_types:
        arith_results = arith_ext.extract(text)
        if any(not r.metadata.get("satisfied", True) for r in arith_results):
            return True
    return False


def _eval_precision_fp(rows: list[dict], predict_fn: Any) -> dict[str, float | int]:
    tp = fp = tn = fn = 0
    for row in rows:
        flagged = predict_fn(row["step_text"])
        is_wrong = row["label"] == "incorrect"
        if flagged:
            if is_wrong:
                tp += 1
            else:
                fp += 1
        else:
            if is_wrong:
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "precision": precision,
        "fp_rate": fp_rate,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _write_artifact(
        {
            "experiment": "1212_tier1_constraint_addition_v2",
            "status": "in_progress",
            "tier1_online_addition_honest_verdict": False,
            "honest_verdict": "in_progress",
        }
    )

    t0 = time.time()
    run_date = _dt.datetime.now(_dt.UTC).isoformat()

    # -----------------------------------------------------------------------
    # Import constraint extractors
    # -----------------------------------------------------------------------
    from carnot.pipeline.constraint_addition_agent import ConstraintAdditionAgent
    from carnot.pipeline.extract import ArithmeticExtractor, LogicExtractor, NLExtractor

    arith_ext = ArithmeticExtractor()
    logic_ext = LogicExtractor()
    nl_ext = NLExtractor()

    # -----------------------------------------------------------------------
    # Load FoVer corpus
    # -----------------------------------------------------------------------
    all_rows = _load_fover(N_LEARN + N_HOLDOUT, offset=FOVER_OFFSET)
    learn_rows = all_rows[:N_LEARN]
    holdout_rows = all_rows[N_LEARN : N_LEARN + N_HOLDOUT]

    n_learn_wrong = sum(1 for r in learn_rows if r["label"] == "incorrect")
    n_learn_correct = sum(1 for r in learn_rows if r["label"] == "correct")
    n_holdout_wrong = sum(1 for r in holdout_rows if r["label"] == "incorrect")
    n_holdout_correct = sum(1 for r in holdout_rows if r["label"] == "correct")

    # -----------------------------------------------------------------------
    # Learning phase: ConstraintAdditionAgent observes violations
    # -----------------------------------------------------------------------
    agent = ConstraintAdditionAgent(
        wrong_threshold=WRONG_THRESHOLD,
        correct_threshold=CORRECT_THRESHOLD,
    )

    for row in learn_rows:
        fired = _get_violated_types(row["step_text"], arith_ext, logic_ext, nl_ext)
        agent.observe(fired, is_correct=(row["label"] == "correct"))

    high_signal_types = agent.detect_additions()
    n_constraints_added = agent.n_constraints_added

    firing_stats_raw = {
        ctype: {
            "wrong_rate": stats.wrong_rate,
            "correct_rate": stats.correct_rate,
            "is_high_signal": stats.is_high_signal(WRONG_THRESHOLD, CORRECT_THRESHOLD),
            "n_wrong_fired": stats.n_wrong_fired,
            "n_correct_fired": stats.n_correct_fired,
            "n_wrong_total": stats.n_wrong_total,
            "n_correct_total": stats.n_correct_total,
        }
        for ctype, stats in agent.firing_stats().items()
    }

    # -----------------------------------------------------------------------
    # Holdout evaluation: before and after
    # -----------------------------------------------------------------------
    before_stats = _eval_precision_fp(
        holdout_rows,
        lambda t: _predict_before(t, arith_ext, logic_ext, nl_ext),
    )
    after_stats = _eval_precision_fp(
        holdout_rows,
        lambda t: _predict_after(t, high_signal_types, arith_ext),
    )

    precision_before = before_stats["precision"]
    precision_after = after_stats["precision"]
    fp_rate_before = before_stats["fp_rate"]
    fp_rate_after = after_stats["fp_rate"]
    precision_improvement = precision_after - precision_before

    beats_reweighting = precision_improvement > EXP134_REWEIGHTING_BASELINE

    if n_constraints_added == 0:
        honest_verdict = "insufficient_patterns_detected"
    elif precision_improvement > 0.01:
        honest_verdict = "constraint_addition_improves_precision"
    elif precision_improvement < -0.01:
        honest_verdict = "constraint_addition_degrades"
    else:
        honest_verdict = "constraint_addition_no_improvement"

    duration_s = round(time.time() - t0, 2)

    # -----------------------------------------------------------------------
    # Write artifact
    # -----------------------------------------------------------------------
    artifact = {
        "experiment": "1212_tier1_constraint_addition_v2",
        "run_date": run_date,
        "status": "complete",
        "duration_s": duration_s,
        # Corpus metadata
        "n_learn_total": N_LEARN,
        "n_learn_wrong": n_learn_wrong,
        "n_learn_correct": n_learn_correct,
        "n_holdout_total": N_HOLDOUT,
        "n_holdout_wrong": n_holdout_wrong,
        "n_holdout_correct": n_holdout_correct,
        # Learning phase results
        "firing_stats": firing_stats_raw,
        "high_signal_types_detected": high_signal_types,
        "wrong_threshold": WRONG_THRESHOLD,
        "correct_threshold": CORRECT_THRESHOLD,
        # Required artifact fields
        "exp134_reweighting_baseline_improvement": EXP134_REWEIGHTING_BASELINE,
        "n_constraints_added": n_constraints_added,
        "precision_before_addition": precision_before,
        "precision_after_addition": precision_after,
        "false_positive_rate_before": fp_rate_before,
        "false_positive_rate_after": fp_rate_after,
        "precision_improvement": precision_improvement,
        "beats_reweighting_baseline": beats_reweighting,
        "tier1_online_addition_honest_verdict": True,
        "honest_verdict": honest_verdict,
        # Detail stats
        "before_tp": before_stats["tp"],
        "before_fp": before_stats["fp"],
        "before_tn": before_stats["tn"],
        "before_fn": before_stats["fn"],
        "after_tp": after_stats["tp"],
        "after_fp": after_stats["fp"],
        "after_tn": after_stats["tn"],
        "after_fn": after_stats["fn"],
    }

    _write_artifact(artifact)

    print(f"Exp 1212 complete in {duration_s}s")
    print(f"  high_signal_types: {high_signal_types}")
    print(f"  n_constraints_added: {n_constraints_added}")
    print(f"  precision_before:  {precision_before:.4f}")
    print(f"  precision_after:   {precision_after:.4f}")
    print(f"  improvement:       {precision_improvement:+.4f}")
    print(f"  fp_rate_before:    {fp_rate_before:.4f}")
    print(f"  fp_rate_after:     {fp_rate_after:.4f}")
    print(f"  honest_verdict:    {honest_verdict}")
    print(f"  beats_reweighting: {beats_reweighting}")


if __name__ == "__main__":
    main()
