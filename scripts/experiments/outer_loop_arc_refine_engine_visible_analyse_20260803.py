"""Analysis + the PRE-COMMITTED stopping-rule verdict for REQ-ARC-WMTE-6091.

THE RULE, RESTATED BEFORE ANY NUMBER IS READ (operator, 2026-08-03): if
refinement-with-the-engine-visible does not beat single-shot induction on gradeable acceptance
cells, the ARC induction line CLOSES. A clean null is a SUCCESS. No follow-up experiment is
authorised on the grounds that the instrument was still imperfect.

WHAT IS PRE-REGISTERED HERE, and it is written down BEFORE the shard is read
--------------------------------------------------------------------------
PRIMARY      held-out `change_accuracy` on the acceptance block. Per game = mean over trials;
             comparison = paired two-sided EXACT SIGN TEST at GAME clustering (games, not
             cells, are the independent unit -- trials of one game share a window).
SECONDARY    `change_fidelity` (continuous, symmetric cell-level union fidelity) and
             `cell_recall`, on the SAME rows. Pre-registered because the primary is coarse:
             every gradeable acceptance block holds exactly ONE changing row at these window
             sizes, so `change_accuracy` is binary per cell and can only move in whole games.
             A null on a metric that could not move is not evidence; reporting the continuous
             metric beside it is what makes the null interpretable.
DENOMINATOR  gradeable games only -- a game counts iff the ORACLE reaches change_accuracy 1.0
             on its acceptance block AND the split reports `decidable`. Excluded games are
             NAMED, with their reason. Missing cells are NAMED, never read as zero.
MIN REACHABLE p  2 * 0.5^G at G discordant games. Reported ALONGSIDE the observed p so a
             "p=1.0" can be distinguished from "no p below X was ever reachable".

THREE COMPARISONS, because two of them are the interpretation of the third:
  treatment vs single_shot   THE STOPPING RULE.
  treatment vs control       the MECHANISM: does showing the engine change anything at all?
                             Paired at the SAMPLE (identical round-0 engine), so this contrast
                             carries no round-0 sampling noise.
  control   vs single_shot   the prior null, re-measured on a gradeable denominator.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
SHARD = REPO / "results" / "exp6091_refine_engine_visible_shard.jsonl"
ARTIFACT = REPO / "results" / "experiment_6091_refine_engine_visible_analysis.json"
SEED = 6091

METRICS = ("change_accuracy", "change_fidelity", "cell_recall")


def sign_test(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """Exact two-sided sign test. `pairs` are (baseline, treatment) at GAME level."""
    wins = sum(1 for b, t in pairs if t > b)
    losses = sum(1 for b, t in pairs if t < b)
    ties = sum(1 for b, t in pairs if t == b)
    n = wins + losses
    if n == 0:
        p: Optional[float] = 1.0
        min_p: Optional[float] = None
    else:
        k = min(wins, losses)
        tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5**n)
        p = min(1.0, 2.0 * tail)
        min_p = min(1.0, 2.0 * (0.5**n))
    return {
        "n_games": len(pairs),
        "wins_treatment": wins,
        "losses_treatment": losses,
        "ties": ties,
        "n_discordant": n,
        "p_two_sided": None if p is None else round(p, 8),
        "min_reachable_p_at_this_n_discordant": None if min_p is None else round(min_p, 8),
        "min_reachable_p_if_all_games_discordant": round(min(1.0, 2.0 * 0.5 ** len(pairs)), 8)
        if pairs
        else None,
    }


def bootstrap_ci(deltas: list[float], n: int = 10000) -> dict[str, Any]:
    """Percentile bootstrap over GAMES (the clustering unit), not over cells."""
    if not deltas:
        return {"mean": None, "ci95": None}
    rng = random.Random(SEED)
    means = []
    for _ in range(n):
        s = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        means.append(sum(s) / len(s))
    means.sort()
    return {
        "mean": round(sum(deltas) / len(deltas), 6),
        "ci95": [
            round(means[int(0.025 * n)], 6),
            round(means[min(n - 1, int(0.975 * n))], 6),
        ],
        "n_games": len(deltas),
        "n_boot": n,
    }


def cell_value(row: dict[str, Any], arm: str, metric: str) -> Optional[float]:
    """MISSING IS NOT ZERO. A cell that produced no engine yields None and is carried as None
    all the way to the aggregation, where it is COUNTED, not silently averaged away."""
    if arm == "single_shot":
        blk = row.get("single_shot")
        return None if not isinstance(blk, dict) else blk.get(metric)
    blk = row.get(arm)
    if not isinstance(blk, dict):
        return None
    key = {
        "change_accuracy": "best_change_accuracy",
        "change_fidelity": "best_change_fidelity",
        "cell_recall": "best_cell_recall",
    }[metric]
    return blk.get(key)


def main() -> int:
    t0 = time.time()
    if not SHARD.exists():
        print(json.dumps({"honest_verdict": "blocked_shard_absent", "shard": str(SHARD)}))
        return 1
    rows = [json.loads(x) for x in SHARD.read_text().splitlines() if x.strip()]

    # ---- gradeability, decided per GAME from the shard's own recorded oracle control ---------
    per_game_cells: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        per_game_cells.setdefault(r.get("game", "?"), []).append(r)

    gradeable, excluded = [], []
    for game, cells in sorted(per_game_cells.items()):
        good = [c for c in cells if c.get("oracle_reaches_1") and c.get("acceptance_decidable")]
        if good:
            gradeable.append(game)
        else:
            reasons = sorted({str(c.get("acceptance_reason")) for c in cells})
            excluded.append(
                {
                    "game": game,
                    "reason": reasons,
                    "oracle_reaches_1": [c.get("oracle_reaches_1") for c in cells],
                    "n_cells": len(cells),
                }
            )

    # ---- per-game aggregation over trials ----------------------------------------------------
    agg: dict[str, dict[str, dict[str, Any]]] = {}
    missing: list[dict[str, Any]] = []
    for game in gradeable:
        agg[game] = {}
        for metric in METRICS:
            for arm in ("single_shot", "refine_control", "refine_treatment"):
                vals = []
                for c in per_game_cells[game]:
                    if not (c.get("oracle_reaches_1") and c.get("acceptance_decidable")):
                        continue
                    v = cell_value(c, arm, metric)
                    if v is None:
                        missing.append(
                            {"game": game, "trial": c.get("trial"), "arm": arm, "metric": metric}
                        )
                    else:
                        vals.append(float(v))
                agg[game][f"{arm}::{metric}"] = {
                    "mean": round(sum(vals) / len(vals), 6) if vals else None,
                    "n_trials": len(vals),
                    "values": vals,
                }

    # ---- the three comparisons ---------------------------------------------------------------
    comparisons: dict[str, Any] = {}
    for metric in METRICS:
        for label, base_arm, trt_arm in (
            ("treatment_vs_single_shot", "single_shot", "refine_treatment"),
            ("treatment_vs_control", "refine_control", "refine_treatment"),
            ("control_vs_single_shot", "single_shot", "refine_control"),
        ):
            pairs, used, dropped = [], [], []
            for game in gradeable:
                b = agg[game][f"{base_arm}::{metric}"]["mean"]
                t = agg[game][f"{trt_arm}::{metric}"]["mean"]
                if b is None or t is None:
                    dropped.append(game)
                    continue
                pairs.append((float(b), float(t)))
                used.append(game)
            comparisons[f"{metric}::{label}"] = {
                "metric": metric,
                "baseline_arm": base_arm,
                "treatment_arm": trt_arm,
                "games_used": used,
                "games_dropped_incomplete": dropped,
                "per_game_delta": {g: round(t - b, 6) for g, (b, t) in zip(used, pairs)},
                "sign_test": sign_test(pairs),
                "bootstrap_delta": bootstrap_ci([t - b for b, t in pairs]),
                "pooled_baseline_mean": round(sum(b for b, _ in pairs) / len(pairs), 6)
                if pairs
                else None,
                "pooled_treatment_mean": round(sum(t for _, t in pairs) / len(pairs), 6)
                if pairs
                else None,
            }

    # ---- stratification by changed-cell count (failure mode 7) -------------------------------
    strata: dict[str, list[Any]] = {"1_cell": [], "2_to_9_cells": [], "10_plus_cells": []}
    for r in rows:
        counts = r.get("acceptance_changed_cells_per_row") or []
        for c in counts:
            key = "1_cell" if c == 1 else ("2_to_9_cells" if c < 10 else "10_plus_cells")
            strata[key].append({"game": r.get("game"), "changed_cells": c})
    strat_summary = {
        k: {"n_rows": len(v), "games": sorted({x["game"] for x in v})} for k, v in strata.items()
    }

    # ---- controls, and whether each is VACUOUS -----------------------------------------------
    ident_acc = [r.get("identity_acceptance", {}) for r in rows]
    ident_full = [r.get("identity_full_window", {}) for r in rows]
    noop_on_acceptance = [int(b.get("n_noop") or 0) for b in ident_acc if isinstance(b, dict) and b]
    controls = {
        "identity_on_acceptance": {
            "change_accuracy_values": sorted(
                {b.get("change_accuracy") for b in ident_acc if isinstance(b, dict)} - {None}
            ),
            "n_noop_values": sorted(set(noop_on_acceptance)),
            "VACUOUS": all(x == 0 for x in noop_on_acceptance) if noop_on_acceptance else None,
            "why": (
                "Every gradeable acceptance row CHANGES and n_noop is 0, so an identity engine "
                "scores 0.0 BY CONSTRUCTION on this block. This control carries no information "
                "here and is reported as vacuous rather than as evidence."
            ),
        },
        "identity_on_full_window": {
            "change_accuracy_values": sorted(
                {b.get("change_accuracy") for b in ident_full if isinstance(b, dict)} - {None}
            ),
            "n_noop_values": sorted(
                {int(b.get("n_noop") or 0) for b in ident_full if isinstance(b, dict)}
            ),
            "why": "Run because the acceptance-block identity control is vacuous; the full "
            "window contains no-op rows, so this control CAN move.",
        },
        "oracle_on_acceptance": {
            "reaches_1_by_game": {
                g: sorted({bool(c.get("oracle_reaches_1")) for c in cs})
                for g, cs in sorted(per_game_cells.items())
            }
        },
    }

    # ---- purity -------------------------------------------------------------------------------
    leaks = sum(int((r.get("acceptance_purity") or {}).get("n_leaks") or 0) for r in rows)
    delivery = {
        "n_cells": len(rows),
        "total_acceptance_answer_leaks_into_any_prompt": leaks,
        "treatment_prompts_containing_engine": sum(
            int((r.get("refine_treatment") or {}).get("n_prompts_with_engine") or 0)
            for r in rows
            if isinstance(r.get("refine_treatment"), dict)
        ),
        "control_prompts_containing_engine": sum(
            int((r.get("refine_control") or {}).get("n_prompts_with_engine") or 0)
            for r in rows
            if isinstance(r.get("refine_control"), dict)
        ),
    }

    # ---- THE VERDICT --------------------------------------------------------------------------
    primary = comparisons.get("change_accuracy::treatment_vs_single_shot", {})
    st = primary.get("sign_test", {})
    beat = bool(
        st.get("wins_treatment", 0) > st.get("losses_treatment", 0)
        and (st.get("p_two_sided") is not None and st["p_two_sided"] < 0.05)
    )
    sec = comparisons.get("change_fidelity::treatment_vs_single_shot", {})
    sec_st = sec.get("sign_test", {})
    sec_beat = bool(
        sec_st.get("wins_treatment", 0) > sec_st.get("losses_treatment", 0)
        and (sec_st.get("p_two_sided") is not None and sec_st["p_two_sided"] < 0.05)
    )

    verdict_line = (
        "REFINEMENT-WITH-ENGINE-VISIBLE BEATS SINGLE-SHOT"
        if beat
        else "NULL: refinement-with-engine-visible does NOT beat single-shot"
    )

    out = {
        "experiment": "experiment_6091_refine_engine_visible_analysis",
        "spec": "REQ-ARC-WMTE-6091",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "random_seed": SEED,
        "cited_upstream_artifacts": [
            {
                "experiment_id": "exp6091_shard",
                "path": str(SHARD.relative_to(REPO)),
                "sha256": hashlib.sha256(SHARD.read_bytes()).hexdigest(),
                "n_cells": len(rows),
            }
        ],
        "n_cells": len(rows),
        "gradeable_games": gradeable,
        "n_gradeable_games": len(gradeable),
        "excluded_games": excluded,
        "missing_arm_values": missing,
        "per_game": agg,
        "comparisons": comparisons,
        "stratification_by_changed_cells": strat_summary,
        "controls": controls,
        "purity_and_delivery": delivery,
        "stopping_rule": {
            "statement": (
                "Pre-committed by the operator 2026-08-03 before any number existed: if "
                "refinement-with-the-engine-visible does not beat single-shot on gradeable "
                "cells, the ARC induction line CLOSES and ARC drops to its CLAUDE.md floor "
                "of one slot per milestone."
            ),
            "primary_metric": "change_accuracy on the acceptance block, game-clustered",
            "primary_result": verdict_line,
            "primary_sign_test": st,
            "secondary_change_fidelity_beats": sec_beat,
            "secondary_sign_test": sec_st,
            "line_closes": (not beat),
        },
        "duration_s": round(time.time() - t0, 3),
    }
    out["honest_verdict"] = (
        "complete_refinement_beats_single_shot"
        if beat
        else "complete_null_refinement_does_not_beat_single_shot_line_closes"
    )
    out["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in out.items() if k != "reproducibility_checksum"},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    ARTIFACT.write_text(json.dumps(out, indent=1))
    print(
        json.dumps(
            {
                "n_cells": len(rows),
                "gradeable_games": gradeable,
                "excluded": [e["game"] for e in excluded],
                "primary": primary.get("sign_test"),
                "primary_pooled": [
                    primary.get("pooled_baseline_mean"),
                    primary.get("pooled_treatment_mean"),
                ],
                "secondary_fidelity": sec.get("sign_test"),
                "secondary_pooled": [
                    sec.get("pooled_baseline_mean"),
                    sec.get("pooled_treatment_mean"),
                ],
                "mechanism_trt_vs_ctl_fidelity": comparisons.get(
                    "change_fidelity::treatment_vs_control", {}
                ).get("sign_test"),
                "leaks": leaks,
                "verdict": out["honest_verdict"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
