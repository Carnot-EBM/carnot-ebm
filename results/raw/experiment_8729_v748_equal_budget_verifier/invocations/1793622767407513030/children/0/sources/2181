"""Experiment 6214: early-stop-grace counterfactual A/B, replayed against real recorded data.

WHY THIS IS NOT A LIVE RE-RUN. The grace mechanism (StepwiseExplorer.is_done(),
arc_competition_agent.py:4439-4449) makes exactly one decision: once at least one level has been
reached, if `early_stop_grace` actions pass with no NEW level-up, stop. It does not change any
exploration behavior BEFORE that decision point -- the agent's action sequence up to the moment
grace would fire is identical regardless of the grace value. That means a recorded run with grace
OFF (so nothing was ever cut short) already contains the exact ground truth needed to compute,
for any candidate grace value G, whether that G would have cut the run short and what it would
have cost. Replaying the existing null-grace rows in results/early_stop_sweep_20260726/rows_*.json
against candidate G values is therefore an EXACT counterfactual, not an approximation -- and it
needs no new game runs, no GPU, and no wall-clock cost beyond reading JSON.

Spec ref: REQ-ARC-WMTE-6220 (the safety revert this experiment follows up on).

Origin: 2026-08-08 operator directive ("yes" to running the grace A/B), after an ad-hoc audit in
this session's own chat found tu93 and cd82 both hit inter-level-up gaps exceeding 400 actions.
This experiment formalizes that audit across the FULL public-game roster (25 games x 3 seeds,
not just the 12 games spot-checked in chat) and sweeps multiple candidate grace values instead of
only checking whether 400 is broken.

METHODOLOGY. For each (game, seed) row at budget=2000 with early_stop_grace=None (so
`level_up_actions` and `actions` are the TRUE, uncut values): for each candidate grace value G,
walk the recorded level-up action indices L[0..n-1]. Before L[0], grace cannot fire (no level
reached yet). For each consecutive pair, if L[i+1]-L[i] > G, the simulated run stops at L[i]+G,
banking i+1 levels and never reaching L[i+1] -- this is the DOWNSIDE (a real level lost). If every
transition survives, and the tail after the last level-up (actions - L[n-1]) exceeds G, the
simulated run stops at L[n-1]+G with ALL n levels still banked -- this is the UPSIDE (wall-clock
saved, and per the is_done() code's own 2026-07-26 correction comment, a not-yet-completed tail
scores 0 regardless of length, so cutting it costs nothing). Rows with n=0 (no level ever reached)
are unaffected by any G (grace only activates once >=1 level is banked).
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT / "python", REPO_ROOT):
    if str(_p) not in sys.path:  # pragma: no cover - direct script guard
        sys.path.insert(0, str(_p))

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_6214_early_stop_grace_counterfactual_ab"
RESULT_RELATIVE_PATH = "results/experiment_6214_early_stop_grace_counterfactual_ab.json"
SCHEMA = "carnot.exp6214.early_stop_grace_counterfactual_ab.v1"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

SWEEP_GLOB = "results/early_stop_sweep_20260726/rows_*.json"
TARGET_BUDGET = 2000  # the shipped MAX_ACTIONS as of this session
CANDIDATE_GRACE_VALUES: tuple[Optional[int], ...] = (None, 400, 800, 1204, 1300, 1600, 2000)


def _load_null_grace_rows(root: Path = REPO_ROOT) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for fpath in sorted(glob.glob(str(root / SWEEP_GLOB))):
        d = json.loads(Path(fpath).read_text(encoding="utf-8"))
        for r in d.get("rows", []):
            if r.get("early_stop_grace") is not None:
                continue
            if r.get("budget") != TARGET_BUDGET:
                continue
            rows.append(
                {
                    "game": r.get("game"),
                    "seed": r.get("seed"),
                    "source_file": Path(fpath).name,
                    "level_up_actions": list(r.get("level_up_actions") or []),
                    "actions": r.get("actions"),
                    "levels": r.get("levels"),
                }
            )
    return rows


def _simulate_one(row: JsonDict, grace: Optional[int]) -> JsonDict:
    """Replay one null-grace row's TRUE level-up sequence against a candidate grace value.

    Returns the simulated (levels_reached, actions_used, stopped_by_grace, cost_a_level)."""
    ups: list[int] = row["level_up_actions"]
    true_actions: int = row["actions"]
    if grace is None or not ups:
        # grace off, or the run never reached a level at all -- grace cannot fire either way.
        return {
            "levels_reached": row["levels"],
            "actions_used": true_actions,
            "stopped_by_grace": False,
            "cost_a_level": False,
        }
    for i in range(len(ups) - 1):
        gap = ups[i + 1] - ups[i]
        if gap > grace:
            return {
                "levels_reached": i + 1,
                "actions_used": ups[i] + grace,
                "stopped_by_grace": True,
                "cost_a_level": (i + 1) < row["levels"],
            }
    tail = true_actions - ups[-1]
    if tail > grace:
        return {
            "levels_reached": len(ups),
            "actions_used": ups[-1] + grace,
            "stopped_by_grace": True,
            "cost_a_level": False,  # the tail is, by construction, not a completed level
        }
    return {
        "levels_reached": row["levels"],
        "actions_used": true_actions,
        "stopped_by_grace": False,
        "cost_a_level": False,
    }


def build_artifact(root: Path = REPO_ROOT) -> JsonDict:
    rows = _load_null_grace_rows(root)
    games = sorted({r["game"] for r in rows})

    per_grace: dict[str, JsonDict] = {}
    for grace in CANDIDATE_GRACE_VALUES:
        key = "off" if grace is None else str(grace)
        sims = [dict(r, sim=_simulate_one(r, grace)) for r in rows]
        levels_lost_rows = [s for s in sims if s["sim"]["cost_a_level"]]
        per_grace[key] = {
            "grace": grace,
            "n_rows": len(sims),
            "total_levels_reached": sum(s["sim"]["levels_reached"] for s in sims),
            "total_actions_used": sum(s["sim"]["actions_used"] for s in sims),
            "n_rows_stopped_by_grace": sum(1 for s in sims if s["sim"]["stopped_by_grace"]),
            "n_rows_that_lost_a_real_level": len(levels_lost_rows),
            "games_that_lost_a_level": sorted({s["game"] for s in levels_lost_rows}),
            "per_row_detail": [
                {
                    "game": s["game"],
                    "seed": s["seed"],
                    "levels_reached": s["sim"]["levels_reached"],
                    "actions_used": s["sim"]["actions_used"],
                    "cost_a_level": s["sim"]["cost_a_level"],
                }
                for s in sims
                if s["sim"]["cost_a_level"] or s["sim"]["stopped_by_grace"]
            ],
        }

    baseline = per_grace["off"]
    baseline_levels = baseline["total_levels_reached"]
    baseline_actions = baseline["total_actions_used"]

    comparison = []
    for grace in CANDIDATE_GRACE_VALUES:
        key = "off" if grace is None else str(grace)
        entry = per_grace[key]
        comparison.append(
            {
                "grace": grace,
                "total_levels_reached": entry["total_levels_reached"],
                "levels_delta_vs_off": entry["total_levels_reached"] - baseline_levels,
                "total_actions_used": entry["total_actions_used"],
                "actions_saved_vs_off": baseline_actions - entry["total_actions_used"],
                "games_that_lost_a_level": entry["games_that_lost_a_level"],
            }
        )

    # The recommended value: the largest candidate that costs ZERO levels vs baseline, for the
    # biggest defensible action-savings. If every non-zero candidate costs a level, recommend off.
    zero_cost = [c for c in comparison if c["grace"] is not None and c["levels_delta_vs_off"] == 0]
    recommended = max(zero_cost, key=lambda c: c["actions_saved_vs_off"]) if zero_cost else None

    honest_verdict = (
        f"complete: grace_{recommended['grace']}_recommended_saves_{recommended['actions_saved_vs_off']}_actions_zero_level_cost"
        if recommended
        else "complete: no_candidate_grace_value_avoids_level_cost_recommend_off"
    )

    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": honest_verdict,
        "target_budget": TARGET_BUDGET,
        "candidate_grace_values": list(CANDIDATE_GRACE_VALUES),
        "games_covered": games,
        "n_games_covered": len(games),
        "n_rows_total": len(rows),
        "comparison": comparison,
        "recommended_grace": recommended["grace"] if recommended else None,
        "per_grace_detail": per_grace,
        "cited_upstream_artifacts": sorted({r["source_file"] for r in rows}),
        "methodology_note": (
            "Exact replay of recorded level_up_actions sequences from null-grace sweep rows "
            "against candidate grace thresholds; not a live re-run. See module docstring for "
            "why this is exact rather than approximate."
        ),
    }


def main() -> None:  # pragma: no cover - thin CLI wrapper, exercised manually
    artifact = build_artifact()
    out = REPO_ROOT / RESULT_RELATIVE_PATH
    out.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out} -- honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()
