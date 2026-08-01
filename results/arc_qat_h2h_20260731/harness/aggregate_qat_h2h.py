#!/usr/bin/env python3
"""Aggregate the QAT vs Q4_K_M inducer head-to-head into one artifact.

WHY THIS IS A NEW FILE AND NOT exp6021's AGGREGATOR WITH THE ARM NAMES CHANGED.
That aggregator carries ~200 lines of narrative specific to its own comparison: the
`/think` prefix being a Qwen3 control token, Qwen hitting the token limit while gemma
does not, a memorizing-subset analysis, and half a dozen cross-references to prior
qwen-arm wedges. Every one of those statements is FALSE for this comparison -- both
arms here are the same model at different quantisations. Renaming the keys would have
produced an artifact whose numbers were right and whose prose asserted things that were
never measured, which is the stale-narrative contamination this project keeps finding.

The statistics are NOT reinvented: `binom_two_sided_sign` is copied verbatim, and the
decisive criterion (coverage + zero-imputed per-game mean, exact two-sided sign test
over the 13 paired games) is the same one exp6021 fixed BEFORE unblinding.

Criterion, fixed before looking at the numbers:
  - primary   : mean-B (zero-imputed held-out mean) per game, sign test over 13 games
  - secondary : coverage (games with any non-zero held-out score)
  - the bar the original cleared was 11-0-2, p = 0.00098
"""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
D = HERE.parent
ARMS = ["q4km", "qat"]
LABEL = {
    "q4km": "gemma-4-31B-it Q4_K_M (non-QAT, shipped)",
    "qat": "gemma-4-31B-it QAT UD-Q4_K_XL",
}


def rows(arm: str) -> list[dict[str, Any]]:
    p = D / f"h2h_shard_{arm}.jsonl"
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            with contextlib.suppress(json.JSONDecodeError):
                out.append(json.loads(line))
    return out


def binom_two_sided_sign(n_pos: int, n_neg: int) -> tuple[float, float, int]:
    """Exact two-sided sign test on discordant pairs -> (p, min_reachable_p, d).

    Copied verbatim from the exp6021 aggregator. Note `min_reachable_p`: with 13 games
    the smallest attainable p is 2/2^d, so a null result must be read against what the
    design COULD have detected, not against 0.05 in the abstract.
    """
    d = n_pos + n_neg
    if d == 0:
        return 1.0, 1.0, 0
    k = min(n_pos, n_neg)
    tail = sum(math.comb(d, i) for i in range(0, k + 1)) / (2**d)
    return round(min(1.0, 2 * tail), 6), round(min(1.0, 2 * (1 / (2**d))), 6), d


def mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 6) if xs else None


def main() -> int:
    data = {a: rows(a) for a in ARMS}
    roster = sorted({r["game"] for r in data["q4km"]} | {r["game"] for r in data["qat"]})

    per_game: dict[str, dict[str, dict[str, Any]]] = {}
    for g in roster:
        per_game[g] = {}
        for a in ARMS:
            cells = [r for r in data[a] if r["game"] == g]
            # A = scorable-only mean (drops cells whose induction failed outright).
            # B = zero-imputed mean (a failed induction scores 0). B is primary: an
            # inducer that fails to produce an importable engine has not "declined to
            # answer", it has answered wrongly, and dropping those cells would reward it.
            scorable = [
                float(r["heldout_accuracy"])
                for r in cells
                if r.get("heldout_accuracy") is not None and r.get("induce_ok")
            ]
            imputed = [
                (
                    float(r["heldout_accuracy"])
                    if r.get("induce_ok") and r.get("heldout_accuracy") is not None
                    else 0.0
                )
                for r in cells
            ]
            per_game[g][a] = {
                "cells": len(cells),
                "induce_ok": sum(1 for r in cells if r.get("induce_ok")),
                "overran": sum(1 for r in cells if r.get("overran")),
                "mean_A_scorable_only": mean(scorable),
                "mean_B_zero_imputed": mean(imputed),
                "mean_cell_recall": mean(
                    [float(r["cell_recall"]) for r in cells if r.get("cell_recall") is not None]
                ),
                "wall_s": round(sum(float(r.get("elapsed_s") or 0) for r in cells), 1),
            }

    # ---- primary: paired sign test on mean-B --------------------------------------
    pos = neg = tie = 0
    per_game_delta = {}
    for g in roster:
        b_ctl = per_game[g]["q4km"]["mean_B_zero_imputed"] or 0.0
        b_trt = per_game[g]["qat"]["mean_B_zero_imputed"] or 0.0
        d = round(b_trt - b_ctl, 6)
        per_game_delta[g] = {"q4km": b_ctl, "qat": b_trt, "delta_qat_minus_q4km": d}
        if d > 0:
            pos += 1
        elif d < 0:
            neg += 1
        else:
            tie += 1
    p_b, p_min_b, d_b = binom_two_sided_sign(pos, neg)

    # ---- secondary: coverage ------------------------------------------------------
    cov = {
        a: sum(1 for g in roster if (per_game[g][a]["mean_B_zero_imputed"] or 0) > 0) for a in ARMS
    }
    cpos = sum(
        1
        for g in roster
        if (per_game[g]["qat"]["mean_B_zero_imputed"] or 0) > 0
        and (per_game[g]["q4km"]["mean_B_zero_imputed"] or 0) == 0
    )
    cneg = sum(
        1
        for g in roster
        if (per_game[g]["q4km"]["mean_B_zero_imputed"] or 0) > 0
        and (per_game[g]["qat"]["mean_B_zero_imputed"] or 0) == 0
    )
    p_c, p_min_c, d_c = binom_two_sided_sign(cpos, cneg)

    pooled = {
        a: mean([float(r["heldout_accuracy"] or 0) if r.get("induce_ok") else 0.0 for r in data[a]])
        for a in ARMS
    }
    induce_rate = {
        a: f"{sum(1 for r in data[a] if r.get('induce_ok'))}/{len(data[a])}" for a in ARMS
    }
    overran = {a: sum(1 for r in data[a] if r.get("overran")) for a in ARMS}
    vram = {
        a: sorted(
            {
                int(r["residency_mib_at_cell_start"])
                for r in data[a]
                if r.get("residency_mib_at_cell_start")
            }
        )[:1]
        for a in ARMS
    }
    wall = {
        a: round(sum(float(r.get("elapsed_s") or 0) for r in data[a]) / 3600.0, 2) for a in ARMS
    }

    decision = (
        "adopt_qat"
        if (p_b < 0.05 and pos > neg)
        else "keep_q4km"
        if (p_b < 0.05 and neg > pos)
        else "indistinguishable_keep_shipped_q4km"
    )

    art = {
        "experiment": "outer_loop_arc_qat_vs_q4km_h2h_20260731",
        "question": (
            "Is unsloth/gemma-4-31B-it-qat-GGUF UD-Q4_K_XL a better ARC world-model inducer "
            "than the shipped unsloth/gemma-4-31B-it-GGUF Q4_K_M? Quantisation is the ONLY "
            "variable; both arms are the same 31B model, same protocol, same 13 games x 3 trials."
        ),
        "decisive_criterion": (
            "Per-game zero-imputed held-out mean (mean-B), exact two-sided sign test over the "
            "13 paired games. Fixed before unblinding, and identical to the criterion exp6021 "
            "used to select gemma-4-31B in the first place."
        ),
        "decision": decision,
        "primary_mean_B": {
            "qat_better_games": pos,
            "q4km_better_games": neg,
            "ties": tie,
            "p_two_sided_sign": p_b,
            "min_reachable_p": p_min_b,
            "discordant_pairs": d_b,
        },
        "secondary_coverage": {
            "nonzero_games": cov,
            "qat_only": cpos,
            "q4km_only": cneg,
            "p_two_sided_sign": p_c,
            "min_reachable_p": p_min_c,
            "discordant_pairs": d_c,
        },
        "pooled_mean_B": pooled,
        "induce_ok_rate": induce_rate,
        "overran_cells": overran,
        "vram_mib_at_cell_start": vram,
        "arm_wall_hours": wall,
        "per_game_delta": per_game_delta,
        "per_game_detail": per_game,
        "labels": LABEL,
        "n_games": len(roster),
        "roster": roster,
        "protocol_provenance": (
            "h2h_arm_runner.py derived from results/inducer_h2h_6021/h2h_arm_runner.py.frozen by "
            "changing ONLY the ARMS dict, then (2026-07-31) parameterising the GPU pin per arm so "
            "the two arms could run concurrently on one RTX 3090 each. n_ctx 32768, budget 16384, "
            "q8_0 KV, ROSTER/TRIALS imported from exp5760 -- all unchanged."
        ),
        "deviation_from_frozen_protocol": (
            "Arms ran CONCURRENTLY on separate GPUs, not sequentially on GPU 1. This run is "
            "therefore equivalent-in-substance to exp6021, NOT byte-identical. Defensible here "
            "because the measured outcome is induction QUALITY on two physically identical RTX "
            "3090s; it would NOT be defensible for a throughput comparison. Each arm still proved "
            "its own card by UUID from residency rather than from CUDA_VISIBLE_DEVICES."
        ),
        "known_log_defect": (
            "The per-arm 'healthy in Ns; residency=N MiB on GPU1' log line hardcodes the string "
            "GPU1 for both arms. The VALUE is correct (computed against each arm's own UUID; the "
            "qat arm's 20428 MiB matches nvidia-smi's GPU 0 reading), but the LABEL is wrong in "
            "the qat log. Cosmetic, recorded rather than silently fixed."
        ),
        "inference_substrate": "live_llm_inference",
        "random_seed": 6021,
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
    }
    out = D.parent / "outer_loop_arc_qat_vs_q4km_h2h_20260731.json"
    out.write_text(json.dumps(art, indent=2) + "\n")

    print(f"=== QAT vs Q4_K_M  ({len(roster)} games x 3 trials) ===")
    print(f"  decision: {decision}")
    print(
        f"  mean-B  : qat better {pos}, q4km better {neg}, ties {tie}  "
        f"-> p={p_b} (min reachable {p_min_b}, d={d_b})"
    )
    print(f"  coverage: {cov}  qat-only {cpos}, q4km-only {cneg}  -> p={p_c}")
    print(f"  pooled mean-B: {pooled}")
    print(f"  induce_ok    : {induce_rate}")
    print(f"  overran      : {overran}")
    print(f"  VRAM MiB     : {vram}")
    print(f"  arm wall (h) : {wall}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
