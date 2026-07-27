#!/usr/bin/env python
"""Analyser over the persisted per-cell row files.

This is an AGGREGATION pass, not a measurement: `inference_substrate` is
`aggregation_from_upstream_artifacts`, and `measurement_wall_s` is summed from each
ROW FILE's own `elapsed_s` -- never from a per-arm wall clock (which undercounts,
because K workers overlap) and never from this analyser's own clock (which is the
analyser clock, not the measurement clock).

Statistics, deliberately conservative:
  * McNemar EXACT (binomial) on the discordant pairs, BOTH tails reported.
  * The MINIMUM REACHABLE p at the observed support, so an underpowered null is
    labelled as underpowered rather than read as evidence of no effect.
  * Paired percentile bootstrap on the rate delta, same method/resamples/seed as the
    baseline artifact's `first_win_ci` (paired_percentile_bootstrap, 1000, 4605) so the
    interval is comparable rather than a differently-defined quantity.
  * Clopper-Pearson exact interval on each arm's own rate.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
OUT = REPO / "results" / "first_win_llm_on_20260727"
CELLS = OUT / "cells"

BASELINE_ARTIFACT = REPO / "results" / "experiment_4605_live_integration_scored_agent.json"


def sha256_file(p: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def load_cells() -> dict[str, dict[str, dict]]:
    by_arm: dict[str, dict[str, dict]] = {}
    for f in sorted(CELLS.glob("*.json")):
        d = json.loads(f.read_text())
        d["_file"] = str(f)
        by_arm.setdefault(d["arm"], {})[d["variant_signature"]] = d
    return by_arm


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float]:
    """Exact binomial interval. No scipy dependency: uses the Beta quantile via
    a bisection on the regularized incomplete beta (math.lgamma-based)."""
    if n == 0:
        return [0.0, 1.0]

    def betainc(a: float, b: float, x: float) -> float:
        # continued-fraction free: simple adaptive Simpson on the Beta pdf, adequate here
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lc = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        n_steps = 20000
        h = x / n_steps
        total = 0.0
        for i in range(n_steps + 1):
            t = i * h
            if t <= 0 or t >= 1:
                val = 0.0 if (a > 1 or b > 1) else 0.0
            else:
                val = math.exp(lc + (a - 1) * math.log(t) + (b - 1) * math.log1p(-t))
            w = 1 if i in (0, n_steps) else (4 if i % 2 else 2)
            total += w * val
        return max(0.0, min(1.0, total * h / 3.0))

    def solve(target: float, a: float, b: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if betainc(a, b, mid) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    low = 0.0 if k == 0 else solve(alpha / 2, k, n - k + 1)
    high = 1.0 if k == n else solve(1 - alpha / 2, k + 1, n - k)
    return [round(low, 6), round(high, 6)]


def mcnemar_exact(b: int, c: int) -> dict:
    """b = pairs where treatment won and control did not; c = the reverse.

    Reports BOTH one-sided tails and the two-sided exact p, plus the minimum p this
    support could possibly have reached (2*0.5^n at n discordant, or 1.0 at n=0). The
    minimum is the honest power statement: if it is above 0.05, no result on this
    support could have been significant, so a p>0.05 says nothing about the effect.
    """
    n = b + c
    if n == 0:
        return {
            "discordant_b_treatment_only": 0,
            "discordant_c_control_only": 0,
            "n_discordant": 0,
            "p_two_sided": 1.0,
            "p_one_sided_treatment_better": 1.0,
            "p_one_sided_control_better": 1.0,
            "min_reachable_p_two_sided_at_this_support": 1.0,
            "significant_at_0_05": False,
            "power_note": (
                "ZERO discordant pairs: every variant agreed between arms. No paired test "
                "can reach any p below 1.0 on zero discordant pairs, so this is an exact "
                "statement of NO OBSERVED per-variant disagreement, not a powered null."
            ),
        }

    def binom_tail_ge(k: int, n_: int) -> float:
        return sum(math.comb(n_, i) for i in range(k, n_ + 1)) / (2.0**n_)

    p_treat = binom_tail_ge(b, n)
    p_ctrl = binom_tail_ge(c, n)
    p_two = min(1.0, 2 * min(p_treat, p_ctrl))
    return {
        "discordant_b_treatment_only": b,
        "discordant_c_control_only": c,
        "n_discordant": n,
        "p_two_sided": round(p_two, 6),
        "p_one_sided_treatment_better": round(p_treat, 6),
        "p_one_sided_control_better": round(p_ctrl, 6),
        "min_reachable_p_two_sided_at_this_support": round(min(1.0, 2 * 0.5**n), 6),
        "significant_at_0_05": bool(p_two < 0.05),
        "power_note": (
            f"{n} discordant pair(s). The smallest two-sided exact p reachable with "
            f"{n} discordant pair(s) is {min(1.0, 2 * 0.5**n):.4g}; "
            + (
                "significance was therefore UNREACHABLE on this support regardless of direction."
                if min(1.0, 2 * 0.5**n) >= 0.05
                else "significance was reachable."
            )
        ),
    }


def paired_bootstrap(
    pairs: list[tuple[bool, bool]], resamples: int = 1000, seed: int = 4605
) -> dict:
    """Paired percentile bootstrap on the rate delta -- the SAME method, resample count and
    seed as the baseline artifact's first_win_ci, so the two intervals are comparable."""
    rng = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return {"method": "paired_percentile_bootstrap", "ci95": [0.0, 0.0], "point": 0.0}
    point = sum(1 for t, c in pairs if t) / n - sum(1 for t, c in pairs if c) / n
    deltas = []
    for _ in range(resamples):
        samp = [pairs[rng.randrange(n)] for _ in range(n)]
        deltas.append(sum(1 for t, c in samp if t) / n - sum(1 for t, c in samp if c) / n)
    deltas.sort()
    lo = deltas[int(0.025 * resamples)]
    hi = deltas[min(resamples - 1, int(0.975 * resamples))]
    return {
        "method": "paired_percentile_bootstrap",
        "bootstrap_resamples": resamples,
        "random_seed": seed,
        "point": round(point, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
    }


def _num(block: dict, key: str) -> int | None:
    """Read a witness counter WITHOUT the falsy-zero trap.

    Returns None when the key is absent/None or carries the shipped -1 "undetermined"
    sentinel; otherwise the integer, with 0 preserved as 0 (never coerced to None or -1).
    A `x or default` idiom cannot express this: `0 or -1` is -1, which is what silently
    disabled the dead-generator check on its own origin-incident cells.
    """
    raw = block.get(key)
    if raw is None:
        return None
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return None
    return None if v < 0 else v


def arm_summary(cells: dict[str, dict]) -> dict:
    sigs = sorted(cells)
    wins = [s for s in sigs if cells[s].get("first_win")]
    errs = [s for s in sigs if cells[s].get("cell_error")]

    # RAW level-up vs REPRODUCED level-up. first_win requires the reproduction gate to pass,
    # so a bare win-count cannot distinguish "the agent never reached a level-up" from "it
    # reached one and the offline replay rejected it". Those have opposite readings, so both
    # are published. claimed_level>0 is the gate's own record of a level-up having occurred.
    def _claimed(s: str) -> int:
        row = cells[s].get("row") or {}
        gate = row.get("reproduction_gate") or {}
        try:
            return int(gate.get("claimed_level") or 0)
        except (TypeError, ValueError):
            return 0

    raw_levelups = [s for s in sigs if _claimed(s) > 0]
    unreproduced = [s for s in raw_levelups if not cells[s].get("first_win")]
    # liveness roll-up. A None here would be a DEAD CHANNEL reading as a clean null, so
    # every field is computed from the witness explicitly and counted, never defaulted.
    lw = [cells[s].get("liveness_witness") or {} for s in sigs]
    llm_blocks = [(w.get("llm") or {}) for w in lw]

    # SENTINEL HANDLING (bug found by smoke-testing this analyser against the real llm_off
    # rows). E3AgentPolicy.generator_liveness_witness writes llm={calls:-1,responses:-1,
    # errors:-1} when the installed proposer has no liveness_witness() method -- which is
    # exactly the _NoOpProposer control arm. -1 means UNDETERMINED, not a count. Summing it
    # produced total_llm_calls=-15 on a 15-cell control arm. Negative values are therefore
    # counted as undetermined and excluded from the totals, never coerced to 0 either (a
    # silent 0 would read as "asked nothing", which is a different and wrong claim).
    def _count(key: str) -> tuple[int, int]:
        total = 0
        undet = 0
        for b in llm_blocks:
            v = _num(b, key)
            if v is None:
                undet += 1
            else:
                total += v
        return total, undet

    calls, calls_undet = _count("calls")
    resp, resp_undet = _count("responses")
    srv_err, _ = _count("errors")
    cont_err, _ = _count("content_failures")
    n_undetermined = max(calls_undet, resp_undet)
    n_ctx_vals = sorted({w.get("generator_n_ctx") for w in lw if w.get("generator_n_ctx")})
    diags: list[str] = []
    for w in lw:
        for d in w.get("generator_server_failure_diagnostics") or []:
            diags.append(str(d)[:300])
    missing_witness = [s for s in sigs if not cells[s].get("liveness_witness")]

    # ERROR CLASS MATTERS, and a bare error COUNT hides it. Two distinct failures appear in
    # these arms and they have opposite readings:
    #   CONTEXT_EXCEEDED   -- the server's own body "Context size has been exceeded.": THE
    #                         concurrency fault under test (pool exhaustion at K x (prompt +
    #                         max_tokens) > n_ctx). Eliminating this is what the fix is for.
    #   REMOTE_DISCONNECTED -- the server closed the connection, i.e. the process went away.
    #                         A SEPARATE fault (the investigation explicitly separated "the
    #                         server survives the overflow" from whatever kills it), and in
    #                         this session it is confounded with an external killer that also
    #                         SIGTERMed two of my own harness processes. Raising n_ctx is not
    #                         expected to remove it and did not.
    # Reporting only the sum would let "the fixed arm still had 16 errors" read as "the fix
    # did not work", when none of those 16 was the fault the fix targets.
    err_classes: dict[str, int] = {}
    for text in diags:
        if "Context size has been exceeded" in text:
            key = "CONTEXT_EXCEEDED"
        elif "RemoteDisconnected" in text:
            key = "REMOTE_DISCONNECTED"
        elif "timed out" in text.lower():
            key = "TIMEOUT"
        else:
            key = "OTHER"
        err_classes[key] = err_classes.get(key, 0) + 1
    return {
        "n_cells": len(sigs),
        "n_first_win": len(wins),
        "first_win_rate": round(len(wins) / len(sigs), 6) if sigs else None,
        "first_win_rate_ci95_clopper_pearson": clopper_pearson(len(wins), len(sigs)),
        "winning_variants": wins,
        "n_raw_levelups_claimed": len(raw_levelups),
        "raw_levelup_variants": raw_levelups,
        "n_levelups_rejected_by_reproduction_gate": len(unreproduced),
        "levelups_rejected_by_reproduction_gate": unreproduced,
        "n_cell_errors": len(errs),
        "cell_errors": errs,
        # Secondary, efficiency-side: ARC's score squares efficiency, so actions-to-first-win
        # matters independently of the win RATE. Reported only over cells that actually won
        # (a non-winner has no actions_to_first_levelup), and the winner set is named so the
        # median is never read as a corpus property when it rests on one cell.
        "actions_to_first_levelup_over_winners": [
            cells[s].get("actions_to_first_levelup") for s in wins
        ],
        "median_actions_to_first_levelup": (
            sorted(
                v for v in (cells[s].get("actions_to_first_levelup") for s in wins) if v is not None
            )[len([1 for s in wins if cells[s].get("actions_to_first_levelup") is not None]) // 2]
            if any(cells[s].get("actions_to_first_levelup") is not None for s in wins)
            else None
        ),
        "median_actions_used_all_cells": round(
            sorted(float(cells[s].get("actions") or 0) for s in sigs)[len(sigs) // 2], 1
        )
        if sigs
        else None,
        "measurement_wall_s_from_rows": round(sum(float(cells[s]["elapsed_s"]) for s in sigs), 2),
        "median_cell_elapsed_s": round(
            sorted(float(cells[s]["elapsed_s"]) for s in sigs)[len(sigs) // 2], 2
        )
        if sigs
        else None,
        "liveness": {
            "n_cells_with_witness": len(sigs) - len(missing_witness),
            "n_cells_missing_witness": len(missing_witness),
            "cells_missing_witness": missing_witness,
            "llm_enabled_cells": sum(1 for w in lw if w.get("llm_enabled")),
            "generator_constructed_cells": sum(1 for w in lw if w.get("generator_constructed")),
            "llm_on_row_valid_cells": sum(1 for w in lw if w.get("llm_on_row_valid")),
            # PROVABLY DEAD: the generator was ASKED (calls>0) and answered NOTHING
            # (responses==0). Guarded against the -1 "undetermined" sentinel so a stub-proposer
            # row can never be scored as dead.
            #
            # DO NOT rewrite these as `int(b.get(k) or -1)`. That is how this exact check broke
            # once already: `responses: 0` is FALSY, so `0 or -1` yields -1, `-1 == 0` is False,
            # and the detector silently stopped firing on the 8 real origin-incident cells it
            # was written for. `_num` returns None for missing/None and preserves 0 as 0.
            "dead_generator_cells": sum(
                1
                for w, b in zip(lw, llm_blocks, strict=True)
                if w.get("llm_enabled")
                and (_num(b, "calls") or 0) > 0
                and _num(b, "responses") == 0
            ),
            "n_cells_llm_counters_undetermined": n_undetermined,
            "llm_counters_undetermined_reasons": sorted(
                {
                    str(w.get("liveness_witness_error"))[:120]
                    for w in lw
                    if w.get("liveness_witness_error")
                }
            ),
            "generator_healthy_after_true_cells": sum(
                1 for w in lw if w.get("generator_healthy_after") is True
            ),
            "generator_healthy_after_false_cells": sum(
                1 for w in lw if w.get("generator_healthy_after") is False
            ),
            "total_llm_calls": calls,
            "total_llm_responses": resp,
            "total_llm_server_errors": srv_err,
            "total_llm_content_failures": cont_err,
            "response_rate": round(resp / calls, 6) if calls else None,
            "generator_n_ctx_observed": n_ctx_vals,
            "n_server_failure_diagnostics": len(diags),
            "server_failure_classes": err_classes,
            "n_context_exceeded_THE_FAULT": err_classes.get("CONTEXT_EXCEEDED", 0),
            "n_remote_disconnected_SEPARATE_FAULT": err_classes.get("REMOTE_DISCONNECTED", 0),
            "server_failure_diagnostics_sample": diags[:6],
            "induction_attempts_total": sum(int(w.get("induction_attempts_n") or 0) for w in lw),
            "induction_attempts_planned_total": sum(
                int(w.get("induction_attempts_planned") or 0) for w in lw
            ),
            # DID THE GENERATOR'S OUTPUT EVER REACH THE POLICY? This is a SEPARATE question
            # from "did the generator answer", and the two can diverge completely: the
            # generator can answer every call cleanly and still contribute nothing, because
            # the induced world model is rejected downstream by a trust/accuracy gate before
            # any plan is installed. Without this channel, a zero first-win delta would be
            # indistinguishable between "the LLM's plans did not help" and "the LLM's plans
            # were never used", which have opposite consequences for what to fix next.
            "n_cells_llm_output_reached_the_policy": sum(
                1 for w in lw if int(w.get("induction_attempts_planned") or 0) > 0
            ),
            "induction_skip_reason_histogram": {
                reason: sum(1 for w in lw if reason in (w.get("induction_attempts_skipped") or []))
                for reason in sorted(
                    {r for w in lw for r in (w.get("induction_attempts_skipped") or []) if r}
                )
            },
        },
    }


def compare(name: str, treat: dict[str, dict], ctrl: dict[str, dict]) -> dict:
    """PER-SEED (per-variant) MATCHED only -- never an any-arm union. Cells present in one
    arm but not the other are reported, not silently dropped into a mismatched N."""
    shared = sorted(set(treat) & set(ctrl))
    only_t = sorted(set(treat) - set(ctrl))
    only_c = sorted(set(ctrl) - set(treat))
    pairs = [(bool(treat[s].get("first_win")), bool(ctrl[s].get("first_win"))) for s in shared]
    b = sum(1 for t, c in pairs if t and not c)
    c_ = sum(1 for t, c in pairs if c and not t)
    tr = sum(1 for t, _ in pairs if t) / len(pairs) if pairs else None
    cr = sum(1 for _, c in pairs if c) / len(pairs) if pairs else None
    return {
        "comparison": name,
        "n_matched_pairs": len(shared),
        "unmatched_treatment_only": only_t,
        "unmatched_control_only": only_c,
        "treatment_first_win_rate": round(tr, 6) if tr is not None else None,
        "control_first_win_rate": round(cr, 6) if cr is not None else None,
        "delta": round(tr - cr, 6) if (tr is not None and cr is not None) else None,
        "flipped_to_win_variants": [
            s for s, (t, c) in zip(shared, pairs, strict=True) if t and not c
        ],
        "flipped_to_loss_variants": [
            s for s, (t, c) in zip(shared, pairs, strict=True) if c and not t
        ],
        "mcnemar_exact": mcnemar_exact(b, c_),
        "paired_bootstrap_delta": paired_bootstrap(pairs),
    }


def main() -> int:
    t0 = time.time()
    by_arm = load_cells()
    if not by_arm:
        print("no cells")
        return 2

    base = json.loads(BASELINE_ARTIFACT.read_text())
    baseline = {
        "path": str(BASELINE_ARTIFACT.relative_to(REPO)),
        "sha256": sha256_file(BASELINE_ARTIFACT),
        "first_win_rate_integrated": base["first_win_rate_integrated"],
        "first_win_rate_bare": base["first_win_rate_bare"],
        "first_win_ci_reported": base["first_win_ci"],
        "n_variant_attempts": base["integrated_measurement"]["variant_attempts_count"],
        "winning_variants": [
            v["variant_signature"]
            for v in base["integrated_measurement"]["variant_attempts"]
            if v.get("first_win")
        ],
        "llm_arm_declared": base["bare_control_config"].get("llm_arm"),
        "inference_substrate": base["inference_substrate"],
        "honest_verdict": base["honest_verdict"],
        "definition": (
            "first_win_rate = (# variant attempts whose reproduction_gate REPRODUCED a "
            "level-up) / (# variant attempts), over the color-permuted held-out variants of "
            "the 25 public games, budget 200 actions, agent = SUBMITTED_AGENT_CONFIG "
            "(E3AgentPolicy, target_levels=SUBMITTED_TARGET_LEVELS, "
            "value_weight=SUBMITTED_VALUE_WEIGHT). Source: "
            "experiment_4605_live_integration_scored_agent.run_variant_attempt -- "
            "first_win == solved == gate.reproduced and reached_level >= claimed >= 1."
        ),
        "baseline_arms_were_llm_off": (
            "experiment_4605._policy_for_mode installs _NoOpProposer for BOTH arms "
            "(line 722), whose induce() returns (False, 'disabled_exp4605_no_live_llm'); "
            "preconditions_checked.live_llm_inference is False. So the 0.04 baseline is an "
            "LLM-OFF measurement taken at concurrency 1."
        ),
    }

    arms = {a: arm_summary(c) for a, c in by_arm.items()}
    comparisons = []
    if "llm_on_fix" in by_arm and "llm_off" in by_arm:
        comparisons.append(
            compare("llm_on_fix_vs_llm_off", by_arm["llm_on_fix"], by_arm["llm_off"])
        )
    if "llm_on_fix" in by_arm and "llm_on_16k" in by_arm:
        comparisons.append(
            compare("llm_on_fix_vs_llm_on_16k_FAULTY", by_arm["llm_on_fix"], by_arm["llm_on_16k"])
        )
    if "llm_on_16k" in by_arm and "llm_off" in by_arm:
        comparisons.append(
            compare("llm_on_16k_FAULTY_vs_llm_off", by_arm["llm_on_16k"], by_arm["llm_off"])
        )

    # CONTROL-WINNER PROBE, kept strictly separate from the pre-specified slice.
    #
    # The pre-specified comparison is the 25-game variant-1 slice, chosen before any arm ran.
    # It is unbiased but it can only detect a LOSS if the control actually wins somewhere in
    # it -- and under today's agent code the control's win set is tiny. So a second, explicitly
    # BIASED-BY-CONSTRUCTION probe is reported alongside it: the cells where the llm_off
    # control DID win. Selecting cells on the control's outcome biases a rate estimate (a
    # control win is the selection criterion, so regression toward no-win is expected), which
    # is exactly why it is NOT pooled with the slice and NOT used as a rate. Its only
    # legitimate reading is directional: does the LLM-on agent HOLD a win the control gets?
    probe = {}
    if "llm_off" in by_arm:
        winners = {s for s, c in by_arm["llm_off"].items() if c.get("first_win")}
        for cond in ("llm_on_fix", "llm_on_16k"):
            # a condition's cells live in its slice arm and/or its *_probe arm
            merged: dict[str, dict] = {}
            merged.update(by_arm.get(cond, {}))
            merged.update(by_arm.get(cond + "_probe", {}))
            shared = sorted(winners & set(merged))
            probe[cond] = {
                "cells": shared,
                "n_cells": len(shared),
                "n_held": sum(1 for s in shared if merged[s].get("first_win")),
                "n_lost": sum(1 for s in shared if not merged[s].get("first_win")),
                "held_cells": [s for s in shared if merged[s].get("first_win")],
                "lost_cells": [s for s in shared if not merged[s].get("first_win")],
                # PER-HELD-CELL LIVENESS (review finding 9). One of the 7 "held" cells in the
                # fixed condition ran with generator_healthy_after=False. Citing a held cell
                # as evidence about the generator, when that cell's own witness says the
                # generator was dead, is the dead-channel reading this measurement exists to
                # refuse -- so the health of each cited cell travels WITH the citation rather
                # than living in a separate arm-level roll-up a reader has to go and find.
                "held_cells_liveness": {
                    s: {
                        "generator_healthy_after": (merged[s].get("liveness_witness") or {}).get(
                            "generator_healthy_after"
                        ),
                        "llm_on_row_valid": (merged[s].get("liveness_witness") or {}).get(
                            "llm_on_row_valid"
                        ),
                        "llm": (merged[s].get("liveness_witness") or {}).get("llm"),
                        "induction_attempts_planned": (merged[s].get("liveness_witness") or {}).get(
                            "induction_attempts_planned"
                        ),
                    }
                    for s in shared
                    if merged[s].get("first_win")
                },
                "n_held_cells_with_unhealthy_generator": sum(
                    1
                    for s in shared
                    if merged[s].get("first_win")
                    and (merged[s].get("liveness_witness") or {}).get("generator_healthy_after")
                    is False
                ),
                "control_won_cells_not_measured_in_this_condition": sorted(winners - set(merged)),
            }
        probe["_what_this_probe_IS"] = (
            "A NEGATIVE CONTROL FOR HARNESS STABILITY -- not a directional check on the "
            "generator. Corrected 2026-07-27 (adversarial review finding 10). It was reported "
            "as the falsifiable complement to the underpowered slice: 'fixed 7/7 held, faulty "
            "7/7 held, with identical actions_to_first_levelup'. But that identity of action "
            "counts is the SIGNATURE OF THE SEVERED PATH -- induction_attempts_planned is 0 on "
            "every cell, so both LLM arms ARE the control agent and 7/7 could not have failed "
            "for any generator-related reason. A check that cannot fail on the variable under "
            "test is not a check on that variable. What it DOES validly show: running the "
            "agent under a live generator at K=4 does not perturb the trajectories the control "
            "produced -- real, and worth having, but a statement about the harness. Its "
            "falsifiability as a generator check is restorable only once "
            "treatment_application[...].treatment_was_applied is True."
        )
        probe["_selection_bias_note"] = (
            "Cells here were selected BECAUSE the control won them. That biases any rate "
            "computed on them toward showing a loss, so no rate or p-value is reported for "
            "this probe -- only the held/lost counts, as a directional check. The unbiased "
            "comparison is the pre-specified variant-1 slice in `comparisons`."
        )
        probe["_control_win_set"] = sorted(winners)

    # ------------------------------------------------------------------ TREATMENT WITNESS
    # THE FATAL FINDING of the 2026-07-27 adversarial review, computed rather than asserted.
    #
    # Every LLM-on arm turned out to be BIT-IDENTICAL to its matched llm_off control on
    # first_win, actions, reached_level AND actions_to_first_levelup. The cause is one field:
    # `induction_attempts_planned` is 0 in every row, i.e. the generator answered and its
    # induced world model was then rejected by a POST-generation trust gate before any plan
    # could be installed. With the generator's output discarded on every cell, the LLM arms
    # ARE the control -- so delta=0, p=1.0 and CI [0,0] are arithmetic identities, not
    # measurements, and no generator state (fixed, faulty or absent) could have moved them.
    #
    # This block is the OUTCOME-LEVEL sensitivity witness the review asked for: it states, at
    # the aggregation level the headline is computed on, whether the pass region for a
    # non-zero delta was reachable at all. `treatment_was_applied` False means the comparison
    # is UNFALSIFIABLE, not underpowered -- more cells would give exactly 0 forever.
    treatment = {}
    ctrl_cells = by_arm.get("llm_off") or {}
    for arm, cells in sorted(by_arm.items()):
        if arm == "llm_off":
            continue
        lw = [c.get("liveness_witness") or {} for c in cells.values()]
        planned = sum(int(w.get("induction_attempts_planned") or 0) for w in lw)
        matched = sorted(set(cells) & set(ctrl_cells))
        fields = ("first_win", "actions", "reached_level", "actions_to_first_levelup")
        diffs = {
            f: sum(1 for s in matched if cells[s].get(f) != ctrl_cells[s].get(f)) for f in fields
        }
        # The gate margins, when the row carries them (added to the witness 2026-07-27). This
        # answers "how far below threshold was it" -- the difference between a gate that a
        # threshold tweak could unblock and one that no tweak could.
        margins = []
        for w in lw:
            for a in w.get("induction_attempt_gate_diagnostics") or []:
                margins.append(a)

        # `rows=margins` binds the loop variable at definition time. Without it ruff's B023
        # fires and, worse, every arm's closure would read whatever `margins` happened to
        # hold when the dict was finally evaluated -- a real late-binding bug, not a style
        # nit, since this closure is called inside the per-arm dict literal below.
        def _vals(key, rows=margins):
            v = [
                a[key]
                for a in rows
                if isinstance(a.get(key), (int, float)) and not isinstance(a.get(key), bool)
            ]
            return {"n": len(v), "min": min(v), "max": max(v)} if v else {"n": 0}

        treatment[arm] = {
            "n_cells": len(cells),
            "induction_attempts_planned_total": planned,
            "treatment_was_applied": planned > 0,
            "n_matched_against_control": len(matched),
            "n_cells_differing_from_matched_control": diffs,
            "arm_is_bit_identical_to_control": all(v == 0 for v in diffs.values()),
            "gate_margins_recorded": len(margins),
            # BOTH GATE BRANCHES. The non-hidden-state branch records verify_accuracy /
            # verify_cell_recall; the HIDDEN_STATE_GAME_IDS branch records trust_energy /
            # heldout_accuracy and IGNORES CARNOT_ARC_TRUST_METRIC entirely. Reporting only
            # the first would leave the 11 hidden-state cells looking uninstrumented -- a
            # dead channel reading as a clean null.
            "verify_accuracy_over_attempts": _vals("verify_accuracy"),
            "verify_cell_recall_over_attempts": _vals("verify_cell_recall"),
            "trust_energy_over_attempts": _vals("trust_energy"),
            "heldout_accuracy_over_attempts": _vals("heldout_accuracy"),
            "heldout_change_consistency_over_attempts": _vals("heldout_change_consistency"),
            "n_attempts_with_no_recorded_margin": sum(
                1
                for a in margins
                if not any(
                    isinstance(a.get(k), (int, float))
                    for k in (
                        "verify_accuracy",
                        "verify_cell_recall",
                        "trust_energy",
                        "heldout_accuracy",
                    )
                )
            ),
            "trust_metric_values": sorted(
                {str(a.get("trust_metric")) for a in margins if a.get("trust_metric")}
            ),
        }
    # THE CROSS-METRIC WITNESS. The trust gate compares ONE of two numbers to 0.5 depending
    # on CARNOT_ARC_TRUST_METRIC. Recording both per attempt makes visible a case the skip
    # REASON string cannot express: an attempt whose accuracy CLEARS the threshold but whose
    # cell_recall does not (lp85: accuracy 0.92, cell_recall 0.0). Under the shipped 'exact'
    # default that attempt passes; under the 'cell_recall' lever it is gated out. So the
    # lever documented as loosening the gate is STRICTER on this corpus, and any claim that
    # "no configuration reaches planned > 0" must be scoped to the configurations actually
    # run rather than generalised.
    disagreements = []
    for arm, cells in sorted(by_arm.items()):
        for sig, c in sorted(cells.items()):
            for a in (c.get("liveness_witness") or {}).get(
                "induction_attempt_gate_diagnostics"
            ) or []:
                acc, cr = a.get("verify_accuracy"), a.get("verify_cell_recall")
                if not (isinstance(acc, (int, float)) and isinstance(cr, (int, float))):
                    continue
                if (acc >= 0.5) != (cr >= 0.5):
                    disagreements.append(
                        {
                            "arm": arm,
                            "cell": sig,
                            "verify_accuracy": acc,
                            "verify_cell_recall": cr,
                            "gating_metric_used": a.get("trust_metric"),
                            "skipped": a.get("skipped"),
                            "planned": a.get("planned"),
                            "would_pass_under_exact": acc >= 0.5,
                            "would_pass_under_cell_recall": cr >= 0.5,
                        }
                    )
    treatment["_metric_disagreements_at_the_0_5_threshold"] = disagreements
    # WHERE ALONG THE PIPELINE DID EACH ATTEMPT DIE? The skip-reason histogram alone reads as
    # "the trust gate rejects everything", but the gate margins show that is not uniformly
    # true: on the shipped 'exact' metric, attempts with verify_accuracy of 0.92-0.96 CLEAR
    # the 0.5 trust threshold and then die further downstream at
    # `no_reachable_plan_after_refinement`. Naming the STAGE matters because the fix differs:
    # a trust-gate rejection is an induction-quality problem, while a
    # no-reachable-plan rejection is a goal/planning problem on a world model the system
    # already trusted.
    stages = {
        "proposer_failed": "1_generation",
        "proposer_failed_or_missing_root": "1_generation",
        "exception": "1_generation",
        "world_model_accuracy_below_threshold": "2_dynamics_trust_gate",
        "hidden_state_trust_below_threshold": "2_dynamics_trust_gate_hidden_state",
        "no_reachable_plan_after_refinement": "3_goal_reachability_AFTER_trust_passed",
        "goal_predicate_unsatisfiable": "3_goal_reachability_AFTER_trust_passed",
    }
    stage_hist: dict = {}
    cleared_trust: list = []
    for arm, cells in sorted(by_arm.items()):
        for sig, c in sorted(cells.items()):
            for a in (c.get("liveness_witness") or {}).get(
                "induction_attempt_gate_diagnostics"
            ) or []:
                stage = stages.get(str(a.get("skipped")), "0_unclassified_or_not_skipped")
                stage_hist.setdefault(arm, {}).setdefault(stage, 0)
                stage_hist[arm][stage] += 1
                if stage.startswith("3_"):
                    cleared_trust.append(
                        {
                            "arm": arm,
                            "cell": sig,
                            "verify_accuracy": a.get("verify_accuracy"),
                            "verify_cell_recall": a.get("verify_cell_recall"),
                            "trust_metric": a.get("trust_metric"),
                            "died_at": a.get("skipped"),
                        }
                    )
    treatment["_where_each_attempt_died_by_stage"] = stage_hist
    treatment["_attempts_that_CLEARED_the_trust_gate_and_died_later"] = cleared_trust
    treatment["_gate_threshold"] = 0.5
    treatment["_what_this_means"] = (
        "treatment_was_applied is False for every arm whose induction_attempts_planned_total "
        "is 0: the LLM tier answered but its output never reached the policy, so that arm is "
        "the control agent under a different label. Any delta, p-value or CI computed between "
        "two such arms is an identity. The correct stamp is UNFALSIFIABLE (the pass region for "
        "a non-zero delta is EMPTY), not 'underpowered' (which invites running more cells, and "
        "more cells of an identity give exactly 0 forever)."
    )

    # ---------------------------------------------------- FIXED-CONDITION LIVENESS ROLL-UP
    # Review finding 9: `n_server_errors_fix == 0` was computed over the 25-cell llm_on_fix
    # arm ALONE while the report's directional claim leans on llm_on_fix_probe, which carries
    # 16 server errors and 2 generator_healthy_after=False cells. A witness must sit at the
    # aggregation level of the claim it supports, so the fixed condition is rolled up across
    # EVERY arm that ran at n_ctx=81920, not just the slice.
    fixed_arms = [a for a in by_arm if a.startswith("llm_on_fix")]
    fx_lw = [c.get("liveness_witness") or {} for a in fixed_arms for c in by_arm[a].values()]
    fixed_rollup = {
        "arms_included": sorted(fixed_arms),
        "n_cells": len(fx_lw),
        "n_server_errors": sum(int((w.get("llm") or {}).get("errors") or 0) for w in fx_lw),
        "n_dead_generator_cells": sum(
            1 for w in fx_lw if w.get("generator_healthy_after") is False
        ),
        "n_rows_llm_on_row_valid": sum(1 for w in fx_lw if w.get("llm_on_row_valid")),
        "scope_note": (
            "THE HONEST SCOPE of 'the fixed arm was server-error free'. It is true of the "
            "25-cell llm_on_fix slice and FALSE of the fixed condition as a whole."
        ),
    }

    runs = {}
    for f in sorted(OUT.glob("run_*.json")):
        runs[f.name] = json.loads(f.read_text())

    payload = {
        "analysis": "arc_first_win_rate_llm_on_at_eval_concurrency",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "baseline": baseline,
        "arms": arms,
        "comparisons": comparisons,
        "control_winner_probe": probe,
        "treatment_application": treatment,
        "fixed_condition_liveness_rollup": fixed_rollup,
        "runs": runs,
        "measurement_wall_s": round(
            sum(a["measurement_wall_s_from_rows"] for a in arms.values()), 2
        ),
        "analyser_wall_s": round(time.time() - t0, 2),
    }
    (OUT / "analysis.json").write_text(json.dumps(payload, indent=1, default=str))
    print(json.dumps({k: v for k, v in payload.items() if k != "runs"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
