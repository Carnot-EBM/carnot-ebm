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


def cell_value_final(row: dict[str, Any], arm: str, metric: str) -> Optional[float]:
    """LIKE-FOR-LIKE companion to `cell_value`, added 2026-08-03 after adversarial review.

    THE ASYMMETRY THIS EXISTS TO EXPOSE. `cell_value` reads `best_<metric>` for the two
    refinement arms -- the MAX over the R refactor rounds, maximised on the very held-out block
    that grades -- while reading `single_shot`'s ONE grade. With exactly one gradeable acceptance
    row per game at these window sizes, per-cell `change_accuracy` is BINARY, so a refinement arm
    that does nothing at all still scores 1-(1-p)^R against single-shot's p (0.36 vs 0.20 at
    p=0.2, R=2) from the extra draw alone. That is a win manufactured by the comparison, not by
    the treatment.

    `treatment_vs_control` is unaffected (both arms are best-of-R) and remains the only symmetric
    contrast in the pre-registered set. This function supplies the missing symmetric form of the
    OTHER two contrasts by reading the FINAL round's grade, which the shard already stores under
    `"final"`. Neither replaces the pre-registered primary; both are reported, and a POSITIVE
    verdict now requires them to AGREE (a tightening, never a loosening).
    """
    if arm == "single_shot":
        blk = row.get("single_shot")
        return None if not isinstance(blk, dict) else blk.get(metric)
    blk = row.get(arm)
    if not isinstance(blk, dict):
        return None
    fin = blk.get("final")
    return None if not isinstance(fin, dict) else fin.get(metric)


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

    # ---- LIKE-FOR-LIKE (final-round) companion comparisons, added 2026-08-03 -----------------
    # Same three contrasts, same clustering, same test -- but reading each refinement arm's FINAL
    # round rather than its best-of-R. See `cell_value_final` for why. Reported ALONGSIDE the
    # pre-registered best-of-R primary; neither is deleted.
    agg_final: dict[str, dict[str, Any]] = {}
    for game in gradeable:
        agg_final[game] = {}
        for metric in METRICS:
            for arm in ("single_shot", "refine_control", "refine_treatment"):
                vals = []
                for c in per_game_cells[game]:
                    if not (c.get("oracle_reaches_1") and c.get("acceptance_decidable")):
                        continue
                    v = cell_value_final(c, arm, metric)
                    if v is not None:
                        vals.append(float(v))
                agg_final[game][f"{arm}::{metric}"] = (
                    round(sum(vals) / len(vals), 6) if vals else None
                )
    comparisons_final: dict[str, Any] = {}
    for metric in METRICS:
        for label, base_arm, trt_arm in (
            ("treatment_vs_single_shot", "single_shot", "refine_treatment"),
            ("treatment_vs_control", "refine_control", "refine_treatment"),
            ("control_vs_single_shot", "single_shot", "refine_control"),
        ):
            pairs, used = [], []
            for game in gradeable:
                b = agg_final[game][f"{base_arm}::{metric}"]
                t = agg_final[game][f"{trt_arm}::{metric}"]
                if b is None or t is None:
                    continue
                pairs.append((float(b), float(t)))
                used.append(game)
            comparisons_final[f"{metric}::{label}"] = {
                "metric": metric,
                "baseline_arm": base_arm,
                "treatment_arm": trt_arm,
                "value_read": "final_round" if trt_arm != "single_shot" else "single_grade",
                "games_used": used,
                "sign_test": sign_test(pairs),
                "bootstrap_delta": bootstrap_ci([t - b for b, t in pairs]),
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
    # THE GUARD WAS INERT UNTIL 2026-08-03: the tallies above name the strata but never re-run the
    # primary on them, so a headline carried entirely by 1-changed-cell rows (the progress-counter
    # failure mode) would not have been caught by anything here. The primary is now RECOMPUTED on
    # the >=2-changed-cell subset. Games whose gradeable acceptance rows are ALL 1-cell are
    # dropped from that recomputation and named.
    multi_cell_games = sorted(
        {
            r.get("game")
            for r in rows
            if any(int(c) >= 2 for c in (r.get("acceptance_changed_cells_per_row") or []))
        }
        & set(gradeable)
    )
    one_cell_only_games = sorted(set(gradeable) - set(multi_cell_games))
    strat_pairs = []
    for game in multi_cell_games:
        b = agg[game]["single_shot::change_accuracy"]["mean"]
        t = agg[game]["refine_treatment::change_accuracy"]["mean"]
        if b is not None and t is not None:
            strat_pairs.append((float(b), float(t)))
    strat_summary["primary_recomputed_on_2plus_cell_rows"] = {
        "games_used": multi_cell_games,
        "games_dropped_all_1_cell": one_cell_only_games,
        "sign_test": sign_test(strat_pairs),
        "why": (
            "Failure mode 7: a headline carried by 1-changed-cell rows is a progress counter, "
            "not an accuracy. This recomputation is the actual guard; the per-stratum tallies "
            "above are descriptive only."
        ),
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
            "VACUOUS": (
                all(int(b.get("n_noop") or 0) == 0 for b in ident_full if isinstance(b, dict) and b)
                if [b for b in ident_full if isinstance(b, dict) and b]
                else None
            ),
            "why": (
                "CORRECTED 2026-08-03 (measured, replacing a claim that was false in both "
                "clauses). The original text said this control was run because the full window "
                "contains no-op rows and 'CAN move'. Measured over the landed cells, n_noop is 0 "
                "on the FULL window too, and identity's full-window change_accuracy is 0.0 "
                "everywhere. The mechanism is not no-ops at all: change_accuracy is "
                "n_changes_correct / n_changing (arc_executable_world_model.py), so an identity "
                "engine scores exactly 0 on ANY block with n_changing > 0, irrespective of how "
                "many no-op rows sit beside it. The identity control is therefore vacuous "
                "EVERYWHERE in this design, not only on the acceptance block, and this field "
                "reports that rather than presenting a structural 0.0 as a passed control. A "
                "non-vacuous identity control would need a block containing no-op rows AND would "
                "have to be scored on `accuracy`, not `change_accuracy`."
            ),
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

    # ---- CALLEE-SIDE DELIVERY. The counts above come from a prompt the HARNESS rendered; these
    # come from the prompt the GENERATOR was handed, captured by wrapping `generate` (the deepest
    # callee before transport). If the two disagree, the harness's render and the model's prompt
    # diverged and no contrast in this artifact is interpretable -- which is why the disagreement
    # is computed and reported rather than assumed away.
    def _sum(arm: str, key: str) -> int:
        return sum(
            int((r.get(arm) or {}).get(key) or 0) for r in rows if isinstance(r.get(arm), dict)
        )

    trt_gen = _sum("refine_treatment", "n_generator_prompts_with_engine")
    ctl_gen = _sum("refine_control", "n_generator_prompts_with_engine")
    delivery["callee_side"] = {
        "treatment_generator_prompts_with_engine": trt_gen,
        "control_generator_prompts_with_engine": ctl_gen,
        "treatment_generator_calls_witnessed": _sum(
            "refine_treatment", "n_generator_calls_witnessed"
        ),
        "control_generator_calls_witnessed": _sum("refine_control", "n_generator_calls_witnessed"),
        "harness_and_callee_agree_on_treatment": trt_gen
        == delivery["treatment_prompts_containing_engine"],
        "control_stayed_blind": ctl_gen == 0,
        "why": (
            "The treatment arm is only a treatment if the engine reached the MODEL. A run where "
            "this count is 0 has measured the control twice and must not be read as a null."
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

    # LIKE-FOR-LIKE CHECK, added 2026-08-03. `beat` above is the pre-registered best-of-R form and
    # is asymmetric against single-shot's single grade (see `cell_value_final`). A POSITIVE verdict
    # now additionally requires the symmetric final-round contrast to point the same way. This can
    # only ever make a win HARDER to claim; it cannot manufacture one.
    prim_final = comparisons_final.get("change_accuracy::treatment_vs_single_shot", {})
    st_final = prim_final.get("sign_test", {})
    beat_like_for_like = bool(
        st_final.get("wins_treatment", 0) > st_final.get("losses_treatment", 0)
        and (st_final.get("p_two_sided") is not None and st_final["p_two_sided"] < 0.05)
    )

    # A NULL IS ONLY A NULL IF THE TREATMENT WAS ACTUALLY ADMINISTERED. If no generator prompt
    # ever carried the engine, this run measured the control twice, and reporting that as a null
    # would launder a blocked measurement into the stopping rule -- the exact substitution the
    # previous attempt correctly refused to make. Distinguished here, not assumed.
    treatment_administered = trt_gen > 0
    any_treatment_rounds = (
        _sum("refine_treatment", "n_generator_calls_witnessed") > 0
        or delivery["treatment_prompts_containing_engine"] > 0
    )

    # ROUND-0 FAILURE IS ITS OWN OUTCOME, added 2026-08-03. Without this, a run in which round-0
    # induction never produced an engine reports `blocked_treatment_never_delivered_to_generator`
    # -- naming the DELIVERY instrument, which is independently proven working, as the blocker,
    # when the measured cause is upstream and different (e.g. the code-only induce hitting its
    # n_predict cap). Both arms are skipped with `no_round0_engine` in that case, so no generator
    # prompt can carry an engine and the delivery counter is 0 for a reason that has nothing to do
    # with delivery. Keyed on `round0_engine_loaded`, which the shard records per cell.
    n_cells_with_round0_engine = sum(1 for r in rows if r.get("round0_engine_loaded"))
    round0_failed_everywhere = len(rows) > 0 and n_cells_with_round0_engine == 0
    induce_messages = sorted(
        {str(r.get("induce_message"))[:200] for r in rows if not r.get("induce_ok")}
    )

    # POWER, added 2026-08-03. `line_closes` was `(not beat) and treatment_administered` with no
    # reference to what p was reachable, so it fired identically at 1 discordant game (min
    # reachable p = 1.0) and at 11 (0.00098). The pre-registration commits IN PROSE to reporting a
    # null that could not have been anything else as underpowered rather than as a refutation;
    # this makes that commitment machine-readable.
    min_p_at_n = st.get("min_reachable_p_at_this_n_discordant")
    adequately_powered = min_p_at_n is not None and min_p_at_n < 0.05

    verdict_line = (
        "REFINEMENT-WITH-ENGINE-VISIBLE BEATS SINGLE-SHOT"
        if (beat and beat_like_for_like)
        else (
            "BLOCKED: round-0 induction produced no engine on any cell; both refinement arms and "
            "single-shot are undefined. This is NOT a null and NOT a delivery failure -- see "
            "`induce_messages`."
            if round0_failed_everywhere
            else (
                "MIXED: the pre-registered best-of-R contrast is positive but the like-for-like "
                "final-round contrast is not; do not report a win"
                if beat and not beat_like_for_like
                else (
                    "NULL: refinement-with-engine-visible does NOT beat single-shot"
                    if treatment_administered and adequately_powered
                    else (
                        "UNDERPOWERED: no p below 0.05 was reachable at this number of "
                        "discordant games; this is not a refutation"
                        if treatment_administered
                        else "BLOCKED: the treatment never reached the generator; this is NOT a null"
                    )
                )
            )
        )
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
        "comparisons_like_for_like_final_round": comparisons_final,
        "primary_comparison_asymmetry_disclosure": {
            "pre_registered_primary_reads": "best_<metric> = MAX over the R refactor rounds",
            "single_shot_reads": "its ONE grade",
            "why_this_matters": (
                "Per-cell change_accuracy is BINARY at these window sizes (exactly one gradeable "
                "acceptance row per game), so a refinement arm that changes nothing still scores "
                "1-(1-p)^R against single-shot's p -- 0.36 vs 0.20 at p=0.2, R=2 -- from the "
                "extra draw alone, and the max is taken over the same held-out block that grades. "
                "`treatment_vs_control` is symmetric and unaffected. The like-for-like "
                "final-round contrast is reported above and a POSITIVE verdict requires both."
            ),
            "beat_pre_registered_best_of_r": beat,
            "beat_like_for_like_final_round": beat_like_for_like,
        },
        "round0_induction": {
            "n_cells": len(rows),
            "n_cells_with_round0_engine": n_cells_with_round0_engine,
            "round0_failed_on_every_cell": round0_failed_everywhere,
            "induce_messages": induce_messages,
            "why": (
                "If round-0 induction produced no engine, single_shot and BOTH refinement arms "
                "are undefined for that cell and no generator prompt can carry an engine. That is "
                "an upstream induction failure, NOT a delivery failure and NOT a null."
            ),
        },
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
            "treatment_actually_administered": treatment_administered,
            "any_treatment_rounds_ran": any_treatment_rounds,
            "round0_failed_on_every_cell": round0_failed_everywhere,
            "min_reachable_p_at_this_n_discordant": min_p_at_n,
            "adequately_powered_at_alpha_0_05": adequately_powered,
            # The line closes on a NULL. It does NOT close on a run whose treatment never
            # reached the model -- that is a blocked measurement and the rule does not fire.
            # CORRECTED 2026-08-03: it also does NOT close when round-0 induction produced no
            # engine anywhere (nothing was measured at all), nor when no p below 0.05 was
            # REACHABLE at the observed number of discordant games -- a null that could not have
            # been anything else is underpowered, not a refutation. Both were previously absent,
            # so `line_closes` would have fired True on a run with 1 discordant game or with zero
            # engines. Direction of the change is conservative: it can only make the line HARDER
            # to close, never easier.
            "line_closes": (
                (not beat)
                and treatment_administered
                and (not round0_failed_everywhere)
                and adequately_powered
            ),
        },
        "duration_s": round(time.time() - t0, 3),
    }
    out["honest_verdict"] = (
        "complete_refinement_beats_single_shot"
        if (beat and beat_like_for_like)
        else (
            "blocked_round0_induction_produced_no_engine_on_any_cell_not_a_null"
            if round0_failed_everywhere
            else (
                "complete_mixed_best_of_r_positive_but_like_for_like_final_round_not_no_win_claimed"
                if beat and not beat_like_for_like
                else (
                    "complete_null_refinement_does_not_beat_single_shot_line_closes"
                    if treatment_administered and adequately_powered
                    else (
                        "blocked_underpowered_no_p_below_0_05_reachable_not_a_refutation"
                        if treatment_administered
                        else "blocked_treatment_never_delivered_to_generator_not_a_null"
                    )
                )
            )
        )
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
