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
                "control_won_cells_not_measured_in_this_condition": sorted(winners - set(merged)),
            }
        probe["_selection_bias_note"] = (
            "Cells here were selected BECAUSE the control won them. That biases any rate "
            "computed on them toward showing a loss, so no rate or p-value is reported for "
            "this probe -- only the held/lost counts, as a directional check. The unbiased "
            "comparison is the pre-specified variant-1 slice in `comparisons`."
        )
        probe["_control_win_set"] = sorted(winners)

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
