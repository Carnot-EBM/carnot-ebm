#!/usr/bin/env python
"""Analysis + artifact for the equalized-budget REAL-holdout induction A/B (exp6474).

Pairs the two arms' shards on (game, trial) and reports, per arm and paired:
  * REAL holdout accuracy (window tail + same-level out-of-window trajectory rows),
    with induce failures treated BOTH ways: imputed as 0.0 (a failed induce is a
    real outcome of the arm) and excluded (capability-given-success). Both are
    reported so neither imputation choice hides in a single number.
  * memorization rate (hardcoded-coordinate scan vs the visible rows).
  * visible-row fit, reported SEPARATELY from holdout so the in-sample number can
    never again be read as generalization.
  * Wilcoxon signed-rank on per-game deltas (exp5726's `_wilcoxon_on_deltas`).

`--early` prints the paired read on whatever cells exist and writes nothing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))


def load_shard(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out[(r["game"], int(r["trial"]))] = r
    return out


def hold_acc(row: dict[str, Any]) -> Optional[float]:
    """Holdout accuracy for a SUCCESSFUL cell; None when the induce failed or the
    engine was unscoreable."""
    if not row.get("induce_ok"):
        return None
    h = row.get("holdout") or {}
    v = h.get("accuracy")
    return float(v) if isinstance(v, (int, float)) else None


def _mean(xs: list[float]) -> Optional[float]:
    return round(statistics.mean(xs), 4) if xs else None


def arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    ok = [r for r in rows if r.get("induce_ok")]
    scored = [r for r in ok if hold_acc(r) is not None]
    imputed = [hold_acc(r) if hold_acc(r) is not None else 0.0 for r in rows]
    mem = [r for r in ok if r.get("is_memorizing")]
    vis = [
        float((r.get("visible_fit") or {}).get("accuracy"))
        for r in ok
        if isinstance((r.get("visible_fit") or {}).get("accuracy"), (int, float))
    ]
    rec = [
        float((r.get("holdout") or {}).get("cell_recall"))
        for r in scored
        if isinstance((r.get("holdout") or {}).get("cell_recall"), (int, float))
    ]
    return {
        "n_cells": n,
        "induce_ok": len(ok),
        "holdout_accuracy_mean_imputed0": _mean(imputed),
        "holdout_accuracy_mean_successes_only": _mean([hold_acc(r) for r in scored]),
        "holdout_cell_recall_mean_successes_only": _mean(rec),
        "memorization_rate_of_successes": round(len(mem) / len(ok), 4) if ok else None,
        "visible_fit_mean_successes_only": _mean(vis),
    }


def paired(
    single: dict[tuple[str, int], dict[str, Any]],
    tool: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    keys = sorted(set(single) & set(tool))
    cells = []
    for k in keys:
        s, t = single[k], tool[k]
        sa = hold_acc(s) if hold_acc(s) is not None else 0.0
        ta = hold_acc(t) if hold_acc(t) is not None else 0.0
        cells.append(
            {
                "game": k[0],
                "trial": k[1],
                "single_holdout": round(sa, 4),
                "tool_holdout": round(ta, 4),
                "delta_tool_minus_single": round(ta - sa, 4),
                "single_induce_ok": bool(s.get("induce_ok")),
                "tool_induce_ok": bool(t.get("induce_ok")),
                "single_memorizing": bool(s.get("is_memorizing")),
                "tool_memorizing": bool(t.get("is_memorizing")),
                "n_holdout": s.get("n_holdout"),
            }
        )
    deltas = [c["delta_tool_minus_single"] for c in cells]
    games = sorted({c["game"] for c in cells})
    per_game = []
    for g in games:
        ds = [c["delta_tool_minus_single"] for c in cells if c["game"] == g]
        per_game.append({"game": g, "delta": round(statistics.mean(ds), 4), "n_trials": len(ds)})
    wilcoxon: dict[str, Any] = {}
    try:
        from carnot.experiment_5726_thinkingcap_16k_dualgpu_reason_ab import (
            _wilcoxon_on_deltas,
        )

        wilcoxon = _wilcoxon_on_deltas({"per_game": per_game})
    except Exception as exc:  # noqa: BLE001 - stats availability must not kill the report
        wilcoxon = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    return {
        "n_paired": len(cells),
        "per_cell": cells,
        "per_game_delta": per_game,
        "pooled_delta_tool_minus_single": _mean(deltas),
        "tool_wins": sum(1 for d in deltas if d > 0),
        "single_wins": sum(1 for d in deltas if d < 0),
        "ties": sum(1 for d in deltas if d == 0),
        "wilcoxon_per_game": wilcoxon,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--early", action="store_true")
    ap.add_argument("--out", help="artifact path")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    single = load_shard(run_dir / "shard_single.jsonl")
    tool = load_shard(run_dir / "shard_tool.jsonl")

    report = {
        "single_arm": arm_summary(list(single.values())),
        "tool_arm": arm_summary(list(tool.values())),
        "paired": paired(single, tool),
    }
    print(json.dumps(report, indent=2))
    if args.early:
        return 0

    meta_s = json.loads((run_dir / "meta_single.json").read_text())
    meta_t = json.loads((run_dir / "meta_tool.json").read_text())
    checksum = hashlib.sha256(
        (
            (run_dir / "shard_single.jsonl").read_text()
            + (run_dir / "shard_tool.jsonl").read_text()
        ).encode()
    ).hexdigest()
    p = report["paired"]
    delta = p["pooled_delta_tool_minus_single"]
    verdict = (
        "complete: equalized-budget real-holdout A/B, "
        f"{p['n_paired']} paired cells; pooled tool-minus-single holdout delta {delta}; "
        f"tool wins {p['tool_wins']}, single wins {p['single_wins']}, ties {p['ties']}"
    )
    artifact = {
        "experiment": 6474,
        "experiment_id": "exp6474-holdout-equalized-induction-ab",
        "title": (
            "Single-shot vs tool-loop induction on a REAL holdout, "
            "equalized evidence and decode budget"
        ),
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inference_substrate": "live_llm_inference",
        "random_seed": 100,
        "reproducibility_checksum": checksum,
        "duration_s": float(meta_s.get("arm_wall_s", 0)) + float(meta_t.get("arm_wall_s", 0)),
        "model_specs": {
            "generator": meta_s.get("hf_id"),
            "gguf": meta_s.get("gguf"),
            "quantisation": meta_s.get("quantisation"),
            "n_ctx": meta_s.get("n_ctx"),
            "kv_quant": meta_s.get("kv_quant"),
            "server_binary": meta_s.get("server_binary"),
            "server_binary_linkage": meta_s.get("server_binary_linkage"),
            "gpu_assignment": {
                "single": {"gpu": meta_s.get("gpu_index"), "uuid": meta_s.get("gpu_uuid")},
                "tool": {"gpu": meta_t.get("gpu_index"), "uuid": meta_t.get("gpu_uuid")},
            },
            "residency_proof": {
                "single_mib": (meta_s.get("server") or {}).get("residency_mib"),
                "tool_mib": (meta_t.get("server") or {}).get("residency_mib"),
            },
        },
        "preconditions_checked": (
            meta_s.get("preconditions_checked", []) + meta_t.get("preconditions_checked", [])
        ),
        "design": {
            "visible_evidence": "holdout_split(window)[0], identical both arms",
            "holdout": (
                "window tail + same-level out-of-window full_traj rows; sizes per game in arm meta"
            ),
            "decode_ceiling_total": 49152,
            "decode_shape": {
                "single": meta_s.get("decode_shape"),
                "tool": meta_t.get("decode_shape"),
            },
            "why_49k_not_102k": (
                "a plain think-on Qwen3.8 induce measures ~41.4k tokens, so 49,152 does "
                "not starve single-shot, and it is the loop's shipped ceiling"
            ),
            "wall_time_disclaimer": (
                "arms ran concurrently on sibling cards sharing CPU/memory bandwidth; "
                "NO wall-time comparison is claimed from this run"
            ),
        },
        "results": report,
        "field_provenance": {
            "results.paired.per_cell.delta_tool_minus_single": {
                "principle": (
                    "Paired same-seed cells difference out the ~40% sampling-noise floor; "
                    "induce failures impute 0.0 because a failed induce is a real outcome "
                    "of the arm, and the successes-only means are reported alongside."
                )
            },
            "results.*.visible_fit_mean_successes_only": {
                "principle": (
                    "The in-sample number, reported separately so it is never again "
                    "conflated with generalization."
                )
            },
            "reproducibility_checksum": {
                "principle": "SHA256 over both shards; any silent row edit changes it."
            },
        },
        "honest_verdict": verdict,
    }
    out = Path(args.out or REPO / "results" / "experiment_6474_holdout_equalized_induction_ab.json")
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n")
    print(f"artifact -> {out}")
    print("verdict:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
