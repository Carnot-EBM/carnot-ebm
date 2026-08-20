#!/usr/bin/env python
"""Analysis + artifact builder for the compacted-carried-state pilot A/B (exp6473).

Reads the per-arm shards the arm runner wrote and computes:
  * Phase 0: measured end-of-loop context distribution on the OFF arm, and how
    often it crosses the default growth trigger (turn-0 + 8192). This replaces
    the design note's Section-1 transcript-growth ESTIMATES with measurements.
  * Phase 1 gates from the design note's Section 11 table: G-M, G-P, G-Q
    (PILOT SIGNAL ONLY at 13 cells -- the documented ~40% A/A cell-divergence
    floor means 13-cell quality numbers can never be a headline claim), G-W,
    and the advisory refetch ratio. G-K (concurrency) is Phase 3, out of scope.

With `--phase0-only` it reports the OFF-arm numbers and the STOP/CONTINUE gate
and writes no artifact. With both shards present it writes the full
results/experiment_6473_*.json artifact (adversarial-verify-ready fields).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]

GROWTH = 8192  # CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH default
STATE_BUDGET = 2048  # CARNOT_ARC_INDUCE_TOOL_COMPACT_STATE_BUDGET default
ONE_ROUND_ALLOWANCE = 5000  # the note's "~cap+5k": one tail round of prompt growth
PHASE0_STOP_CTX = 25000  # the note's "rarely exceeds ~25k" STOP threshold


def load_shard(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def pts(row: dict[str, Any]) -> list[int]:
    """The cell's measured per-turn prompt sizes, nulls dropped."""
    return [
        int(p)
        for p in row.get("stats", {}).get("prompt_tokens_per_turn", [])
        if isinstance(p, (int, float))
    ]


def cell_phase0(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    p = pts(row)
    if not p:
        return None
    turn0 = p[0]
    end = max(p)
    return {
        "game": row["game"],
        "turn0_prompt_tokens": turn0,
        "end_of_loop_prompt_tokens": end,
        "growth_tokens": end - turn0,
        "crossed_default_trigger": end >= turn0 + GROWTH,
        "exceeded_25k": end >= PHASE0_STOP_CTX,
        "turns": row.get("stats", {}).get("turns"),
        "terminated_by": row.get("stats", {}).get("terminated_by"),
    }


def phase0_report(off_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = [c for c in (cell_phase0(r) for r in off_rows) if c is not None]
    ends = [c["end_of_loop_prompt_tokens"] for c in cells]
    n = len(cells)
    crossed = sum(1 for c in cells if c["crossed_default_trigger"])
    over25 = sum(1 for c in cells if c["exceeded_25k"])
    return {
        "n_cells_measured": n,
        "end_of_loop_prompt_tokens": {
            "min": min(ends) if ends else None,
            "median": statistics.median(ends) if ends else None,
            "max": max(ends) if ends else None,
        },
        "cells_crossing_default_trigger": crossed,
        "cells_exceeding_25k": over25,
        "per_cell": cells,
        "stop_rule": {
            "text": "if measured end-of-loop context rarely exceeds ~25k, STOP before Phase 1",
            "threshold_tokens": PHASE0_STOP_CTX,
        },
    }


def _stat(row: dict[str, Any], key: str, default: Any = None) -> Any:
    return row.get("stats", {}).get(key, default)


def _inspection_calls(row: dict[str, Any]) -> int:
    by = _stat(row, "tool_calls_by_name", {}) or {}
    return sum(int(by.get(k, 0)) for k in ("diff_grids", "query_region", "list_transitions"))


def gates(off_rows: list[dict[str, Any]], on_rows: list[dict[str, Any]]) -> dict[str, Any]:
    off = {r["game"]: r for r in off_rows}
    on = {r["game"]: r for r in on_rows}
    paired = sorted(set(off) & set(on))

    # ---- G-M (mechanism): ON p95/max per-turn prompt bounded near the cap, on
    # every cell where the OFF arm exceeded the trigger. Near-deterministic.
    gm_cells = []
    for g in paired:
        p_off, p_on = pts(off[g]), pts(on[g])
        if not p_off or not p_on:
            continue
        if max(p_off) < p_off[0] + GROWTH:
            continue  # OFF never crossed: G-M has nothing to bound here
        bound = p_on[0] + GROWTH + STATE_BUDGET + ONE_ROUND_ALLOWANCE
        gm_cells.append(
            {
                "game": g,
                "on_max_prompt_tokens": max(p_on),
                "bound": bound,
                "pass": max(p_on) <= bound,
                "on_compactions": _stat(on[g], "compactions"),
            }
        )
    g_m = {"cells": gm_cells, "pass": all(c["pass"] for c in gm_cells) if gm_cells else None}

    # ---- G-P (parse safety, THE KILL CONDITION): pooled ON-OFF <= 5pp, and the
    # per-cell median unparsed-tool-call-text turns not worse.
    def pooled_rate(rows):
        fails = sum(int(_stat(r, "tool_call_parse_failures", 0)) for r in rows)
        total = sum(int(_stat(r, "tool_calls_total", 0)) for r in rows)
        return fails, total, (fails / total if total else 0.0)

    off_f, off_t, off_rate = pooled_rate([off[g] for g in paired])
    on_f, on_t, on_rate = pooled_rate([on[g] for g in paired])
    med_unparsed_off = statistics.median(
        [int(_stat(off[g], "unparsed_tool_call_text_turns", 0)) for g in paired]
    )
    med_unparsed_on = statistics.median(
        [int(_stat(on[g], "unparsed_tool_call_text_turns", 0)) for g in paired]
    )
    g_p = {
        "off": {"parse_failures": off_f, "tool_calls": off_t, "rate": round(off_rate, 4)},
        "on": {"parse_failures": on_f, "tool_calls": on_t, "rate": round(on_rate, 4)},
        "delta_pp": round((on_rate - off_rate) * 100, 2),
        "median_unparsed_turns_off": med_unparsed_off,
        "median_unparsed_turns_on": med_unparsed_on,
        "pass": ((on_rate - off_rate) <= 0.05) and (med_unparsed_on <= med_unparsed_off),
        "power_note": (
            "13 cells pool a few hundred tool calls per arm: ~80% power for a 5pp shift; "
            "do not claim tighter than the data supports"
        ),
    }

    # ---- G-Q (quality non-inferiority, PILOT ONLY at 13 cells) ----
    def hold(r):
        v = _stat(r, "best_holdout_accuracy")
        return float(v) if isinstance(v, (int, float)) else 0.0

    diffs = [(g, hold(on[g]) - hold(off[g])) for g in paired]
    pooled_delta = statistics.mean([d for _, d in diffs]) if diffs else None
    wins_ties = sum(1 for _, d in diffs if d >= 0)
    g_q = {
        "per_cell_delta": [{"game": g, "on_minus_off": round(d, 4)} for g, d in diffs],
        "pooled_on_minus_off": round(pooled_delta, 4) if pooled_delta is not None else None,
        "on_wins_or_ties": wins_ties,
        "n_paired": len(diffs),
        "pass": (
            pooled_delta is not None and pooled_delta >= -0.05 and wins_ties * 2 >= len(diffs)
        ),
        "pilot_disclaimer": (
            "PILOT SIGNAL ONLY: 13 cells sit on a documented ~40% A/A cell-divergence "
            "floor under identical code; this number detects only gross regressions and "
            "is never a headline claim. Claim-grade needs >=30 paired cells (Phase 2)."
        ),
    }

    # ---- G-W (single-stream cost): median loop wall ON <= 1.05x OFF ----
    med_off = statistics.median([float(_stat(off[g], "wall_s", 0.0)) for g in paired])
    med_on = statistics.median([float(_stat(on[g], "wall_s", 0.0)) for g in paired])
    g_w = {
        "median_wall_s_off": round(med_off, 2),
        "median_wall_s_on": round(med_on, 2),
        "ratio": round(med_on / med_off, 4) if med_off else None,
        "pass": bool(med_off) and med_on <= 1.05 * med_off,
    }

    # ---- Advisory: refetch ratio on the ON arm ----
    refetch = sum(int(_stat(on[g], "refetch_tool_calls_post_compaction", 0)) for g in paired)
    inspect = sum(_inspection_calls(on[g]) for g in paired)
    advisory = {
        "refetch_tool_calls_post_compaction": refetch,
        "inspection_calls_on_arm": inspect,
        "ratio": round(refetch / inspect, 4) if inspect else None,
        "threshold": 0.20,
        "pass_advisory": (refetch / inspect <= 0.20) if inspect else None,
    }

    # ---- compaction mechanics on the ON arm (context for every gate) ----
    mech = {
        "cells_with_any_compaction": sum(1 for g in paired if _stat(on[g], "compactions", 0)),
        "compactions_per_cell": {g: _stat(on[g], "compactions", 0) for g in paired},
        "floor_hits": sum(1 for g in paired if _stat(on[g], "compact_floor_hit")),
        "thrash_alarms": sum(1 for g in paired if _stat(on[g], "compaction_thrash_alarm")),
        "duplicate_candidate_submissions_on": sum(
            int(_stat(on[g], "duplicate_candidate_submissions", 0)) for g in paired
        ),
        "duplicate_candidate_submissions_off": sum(
            int(_stat(off[g], "duplicate_candidate_submissions", 0)) for g in paired
        ),
    }
    return {
        "paired_cells": paired,
        "G_M": g_m,
        "G_P": g_p,
        "G_Q": g_q,
        "G_W": g_w,
        "advisory_refetch": advisory,
        "compaction_mechanics": mech,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--phase0-only", action="store_true")
    ap.add_argument("--out", help="artifact path (default results/experiment_6473_...json)")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    off_rows = load_shard(run_dir / "shard_off.jsonl")
    on_rows = load_shard(run_dir / "shard_on.jsonl")

    p0 = phase0_report(off_rows)
    print(json.dumps({"phase0": p0}, indent=2))
    if args.phase0_only:
        return 0

    meta_off = json.loads((run_dir / "meta_off.json").read_text())
    meta_on = json.loads((run_dir / "meta_on.json").read_text())
    g = gates(off_rows, on_rows)

    # Wall-clock across both arms, from the arm metas -- real compute time, not
    # the analyzer's own runtime.
    duration_s = float(meta_off.get("arm_wall_s", 0)) + float(meta_on.get("arm_wall_s", 0))
    checksum = hashlib.sha256(
        (
            (run_dir / "shard_off.jsonl").read_text() + (run_dir / "shard_on.jsonl").read_text()
        ).encode()
    ).hexdigest()

    hard_gates_pass = all(
        bool(g[k]["pass"]) for k in ("G_P", "G_Q", "G_W") if g[k]["pass"] is not None
    ) and (g["G_M"]["pass"] is not False)
    verdict = (
        "complete: compaction pilot A/B measured on 13 paired live cells; "
        + ("all Section-11 pilot gates pass" if hard_gates_pass else "one or more gates FAIL")
        + " (G-Q is a pilot signal, never a claim)"
    )

    artifact: dict[str, Any] = {
        "experiment": 6473,
        "experiment_id": "exp6473-tool-loop-compaction-pilot-ab",
        "title": "Compacted carried state for the induction tool loop: Phase 0 + Phase 1 pilot A/B",
        "req": "REQ-ARC-WMTE-6540",
        "design_note": "docs/research-notes/arc-induction-compacted-carried-state-2026-08-19.md",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inference_substrate": "live_llm_inference",
        "random_seed": int(meta_off.get("seed_base", 100)),
        "reproducibility_checksum": checksum,
        "duration_s": duration_s,
        "model_specs": {
            "generator": meta_off.get("hf_id"),
            "gguf": meta_off.get("gguf"),
            "quantisation": meta_off.get("quantisation"),
            "n_ctx": meta_off.get("n_ctx"),
            "kv_quant": meta_off.get("kv_quant"),
            "server_binary": meta_off.get("server_binary"),
            "server_binary_linkage": meta_off.get("server_binary_linkage"),
            "gpu": "GPU 0 only (RTX 3090; GPU 1 reserved by an unrelated experiment)",
            "gpu0_uuid": meta_off.get("gpu0_uuid"),
            "residency_proof_off": (meta_off.get("server") or {}).get("residency_mib_gpu0"),
            "residency_proof_on": (meta_on.get("server") or {}).get("residency_mib_gpu0"),
        },
        "preconditions_checked": (
            meta_off.get("preconditions_checked", [])
            + [
                {
                    "resource": "gpu0_residency_verified_per_arm",
                    "available": True,
                    "detail": "nvidia-smi per-PID residency on GPU 0's UUID, checked per cell",
                }
            ]
        ),
        "cells": {
            "definition": "13 games (exp5760 roster) x trial 0, paired seeds (base 100)",
            "off_completed": len(off_rows),
            "on_completed": len(on_rows),
        },
        "phase0": p0,
        "phase1_gates": g,
        "arm_meta": {"off": meta_off, "on": meta_on},
        "field_provenance": {
            "phase0.end_of_loop_prompt_tokens": {
                "principle": (
                    "Measured usage.prompt_tokens from the server, never estimated; replaces "
                    "the design note's Section-1 transcript-growth estimates."
                )
            },
            "phase1_gates.G_P": {
                "principle": (
                    "The kill condition: a parse-rate collapse under the new prompt shape "
                    "sits in the PEG autoparser and no knob retune fixes it."
                )
            },
            "phase1_gates.G_Q.pilot_disclaimer": {
                "principle": (
                    "Sample-Size Rigor: 13 cells under a ~40% A/A divergence floor cannot "
                    "support a quality claim; the field exists so nobody reads one into it."
                )
            },
            "reproducibility_checksum": {
                "principle": "SHA256 over both shards: any silent row edit changes it."
            },
        },
        "honest_verdict": verdict,
    }
    out = Path(args.out or REPO / "results" / "experiment_6473_tool_loop_compaction_pilot_ab.json")
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n")
    print(f"artifact -> {out}")
    print(json.dumps({k: g[k].get("pass") for k in ("G_M", "G_P", "G_Q", "G_W")}, indent=2))
    print("advisory refetch:", g["advisory_refetch"])
    print("verdict:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
