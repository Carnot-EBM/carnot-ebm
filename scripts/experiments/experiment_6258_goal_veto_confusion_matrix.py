#!/usr/bin/env python3
"""REQ-ARC-WMTE-6258: what does the live goal-predicate veto actually accept and reject?

THE QUESTION. A code-path analysis (ops/known-issues.md 2026-08-12) argued the live veto is
inverted at level 1: `arc_competition_agent` sets `min_goal_predicate_consistency=1.0`,
`arc_llm_reinduction` rejects anything scoring below that, the scored window is the agent's
own transitions, and before its first level-up that window contains no level-ups -- on which
a constant-False predicate scores exactly 1.0. The argument is sound by construction. This
puts a NUMBER on it.

WHAT IS COMPUTED. For every goal predicate whose two axes are already measured, build the
veto's confusion matrix against ground truth:

  * what the veto SEES: specificity accuracy, and whether `accuracy >= 1.0` accepts it.
  * GROUND TRUTH: does the predicate fire on a grid where a level-up really happened?

  FALSE ACCEPT = the veto admits a predicate that never fires on a real win. Those reach
  `plan_in_model` as its termination condition, so the planner cannot terminate or
  terminates on an in-model false win.
  FALSE REJECT = the veto discards a predicate that DOES fire on a real win, because it had
  at least one false positive and so scored below 1.0.

NO NEW MEASUREMENT, NO GPU. Both axes already exist: exp6257 swept the 25 stored engines and
exp6256 induced 8 fresh predicates (control and treatment across 4 games). This is
arithmetic over those artifacts. Re-inducing to answer it would have cost GPU hours for
numbers already on disk.

WHAT THIS CANNOT SHOW. It does not observe the live agent. It shows what the live threshold
WOULD do to predicates of the kind the project actually produces. The population is public
games with development-proxy win grids, not hidden-game level 1.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

OUT = REPO / "results" / "experiment_6258_goal_veto_confusion_matrix.json"
SWEEP = REPO / "results" / "experiment_6257_stored_goal_predicate_sweep.json"
FRESH = REPO / "results" / "experiment_6256_win_exemplar_goal_predicate_ab.json"
# The live value, read off both call sites in arc_competition_agent.py (7164, 7309).
LIVE_THRESHOLD = 1.0


def _collect() -> list[dict]:
    """One row per predicate with BOTH axes present. A predicate missing either axis is
    dropped rather than guessed -- an unmeasurable sensitivity must never read as a pass."""
    rows: list[dict] = []

    if SWEEP.exists():
        for r in json.loads(SWEEP.read_text()).get("per_game_results", []):
            spec, fires = r.get("specificity_accuracy"), r.get("fires_on_real_win")
            if spec is None or fires is None:
                continue
            rows.append(
                {
                    "source": "stored",
                    "game": r["game"],
                    "arm": "stored",
                    "specificity": float(spec),
                    "fires_on_real_win": bool(fires),
                }
            )

    if FRESH.exists():
        for r in json.loads(FRESH.read_text()).get("per_game_results", []):
            for arm in ("control", "treatment"):
                a = r.get(arm)
                if not a:
                    continue
                spec, fires = a.get("goal_specificity_accuracy"), a.get("goal_fires_on_real_win")
                if spec is None or fires is None:
                    continue
                rows.append(
                    {
                        "source": "freshly_induced",
                        "game": r["game"],
                        "arm": arm,
                        "specificity": float(spec),
                        "fires_on_real_win": bool(fires),
                    }
                )
    return rows


def build_artifact() -> dict:
    t0 = time.time()
    rows = _collect()
    for r in rows:
        r["veto_accepts"] = bool(r["specificity"] >= LIVE_THRESHOLD)
        if r["veto_accepts"] and not r["fires_on_real_win"]:
            r["outcome"] = "FALSE_ACCEPT_degenerate_admitted"
        elif not r["veto_accepts"] and r["fires_on_real_win"]:
            r["outcome"] = "FALSE_REJECT_discriminating_discarded"
        elif r["veto_accepts"] and r["fires_on_real_win"]:
            r["outcome"] = "true_accept"
        else:
            r["outcome"] = "true_reject"

    n = len(rows)
    fa = [r for r in rows if r["outcome"] == "FALSE_ACCEPT_degenerate_admitted"]
    fr = [r for r in rows if r["outcome"] == "FALSE_REJECT_discriminating_discarded"]
    ta = [r for r in rows if r["outcome"] == "true_accept"]
    tr = [r for r in rows if r["outcome"] == "true_reject"]
    accepted = fa + ta

    # The number that decides whether the veto is working: of everything it lets through,
    # how much is useless?
    precision = round(len(ta) / len(accepted), 4) if accepted else None

    art = {
        "experiment": "experiment_6258_goal_veto_confusion_matrix",
        "title": "What the live goal-predicate veto accepts and rejects, against ground truth",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "live_threshold": LIVE_THRESHOLD,
        "per_predicate": rows,
        "n_predicates": n,
        "n_true_accept": len(ta),
        "n_false_accept_degenerate_admitted": len(fa),
        "n_false_reject_discriminating_discarded": len(fr),
        "n_true_reject": len(tr),
        "acceptance_precision": precision,
        "acceptance_precision_meaning": (
            "of every predicate the veto ADMITS, the fraction that actually fires on a real "
            "win. The rest are admitted and then used as plan_in_model's termination condition."
        ),
        "false_accept_games": sorted({r["game"] for r in fa}),
        "false_reject_games": sorted({r["game"] for r in fr}),
        "cited_upstream_artifacts": [
            {
                "experiment_id": "exp6257",
                "path": str(SWEEP.relative_to(REPO)),
                "fields_imported": ["specificity_accuracy", "fires_on_real_win"],
            },
            {
                "experiment_id": "exp6256",
                "path": str(FRESH.relative_to(REPO)),
                "fields_imported": ["goal_specificity_accuracy", "goal_fires_on_real_win"],
            },
        ],
        "what_this_cannot_show": (
            "this does not observe the live agent. It shows what the live threshold WOULD do "
            "to predicates of the kind the project actually produces. The population is public "
            "games with development-proxy win grids, not hidden-game level 1."
        ),
        "why_no_new_measurement": (
            "both axes already existed in exp6256 and exp6257. Re-inducing would have spent GPU "
            "hours regenerating numbers already on disk."
        ),
        "solve_provenance": "development_proxy",
        "arc_solve_claim": False,
        "verifier_is_oracle": False,
        "verifier_is_oracle_principle": (
            "this audits a gate against recorded level-up ground truth; it is not the "
            "executable win oracle driving a solve, and no level is claimed"
        ),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "random_seed": 6258,
    }
    if n == 0:
        art["honest_verdict"] = "complete_blocked_no_predicate_had_both_axes_measured"
    else:
        art["honest_verdict"] = (
            f"complete_goal_veto_confusion_{len(fa)}_false_accepts_{len(fr)}_false_rejects_of_{n}_"
            f"acceptance_precision_{precision}"
        )
    art["duration_s"] = round(time.time() - t0, 4)
    payload = {k: v for k, v in art.items() if k != "duration_s"}
    art["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return art


def main() -> int:
    art = build_artifact()
    OUT.write_text(json.dumps(art, indent=2, default=str) + "\n", encoding="utf-8")
    print("verdict:", art.get("honest_verdict"))
    print(
        f"  true_accept={art['n_true_accept']} FALSE_ACCEPT={art['n_false_accept_degenerate_admitted']} "
        f"FALSE_REJECT={art['n_false_reject_discriminating_discarded']} true_reject={art['n_true_reject']}"
    )
    print("  acceptance precision:", art["acceptance_precision"])
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
