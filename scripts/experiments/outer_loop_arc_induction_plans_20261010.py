"""ARC induction measurement on the current stack: plans installed per induce attempt.

Question: with the LLM tier ON and today's pinned generator, how many induce attempts end in
an installed plan? The 2026-07-27 corpus (retired 9B generator) recorded 0 of 136.

This is the July `firstwin.py` method with three changes, so nothing else moves:
  1. No hardcoded generator. The policy builds the SHIPPED proposer (`proposer=None`), so the
     model is whatever the code pins today (Qwen3.8-27B). The server is started through that
     same object, never a hand-built command line.
  2. Paths come from this file's location, not from the main checkout. Output goes to a new
     directory; the July evidence directory is never written.
  3. The card is pinned by PCI bus order, and the server's PID is joined to a GPU UUID after
     the model loads. The UUID is recorded in the run file.

Each cell calls the committed `experiment_4605.run_variant_attempt` unchanged, with the
shipped proposer installed in place of the no-op stub. The cell's liveness witness carries
induction_attempts_n / _planned / _skipped, which are the numbers this measurement needs.

Run (one card only, never the conductor's):
    CARNOT_LLAMA_SERVER=~/.cache/llama.cpp-master/build/bin/llama-server \\
    PYTHONPATH=<worktree>/python python scripts/experiments/outer_loop_arc_induction_plans_20261010.py \\
        --gpu 1 --port 8961 --k 4 --variants 1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO))

# A rerun sets CARNOT_INDUCTION_OUT so it never overwrites the first run's cells.
OUT = REPO / "results" / os.environ.get("CARNOT_INDUCTION_OUT", "arc_induction_plans_20261010")
CELLS = OUT / "cells"
_LOCAL = threading.local()


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()


def device_join(pid: int | None, want_index: int) -> dict:
    """Join the server PID to a GPU UUID from nvidia-smi. Env vars are not evidence."""
    out: dict = {"pid": pid, "want_index": want_index}
    try:
        idx_by_uuid = {}
        for line in _run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]
        ).splitlines():
            i, u = (x.strip() for x in line.split(","))
            idx_by_uuid[u] = int(i)
        rows = []
        for line in _run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid", "--format=csv,noheader"]
        ).splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 3:
                rows.append(
                    {
                        "pid": int(p[0]),
                        "used_mib": int(p[1].split()[0]),
                        "gpu_uuid": p[2],
                        "gpu_index": idx_by_uuid.get(p[2], -1),
                    }
                )
        mine = [r for r in rows if r["pid"] == pid]
        out["all_compute_apps"] = rows
        out["my_rows"] = mine
        out["verdict"] = (
            "CONFIRMED_BY_PID_UUID_JOIN"
            if mine and all(r["gpu_index"] == want_index for r in mine)
            else ("WRONG_DEVICE" if mine else "NOT_RESIDENT_ON_NVIDIA_GPU")
        )
    except Exception as exc:  # a failed join must be visible, never a silent pass
        out["verdict"] = "UNVERIFIED"
        out["error"] = repr(exc)[:200]
    return out


def start_server(exp4605, game: str) -> tuple[object | None, dict]:
    """Start the shipped proposer's server through the policy that will use it."""
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    pol = E3AgentPolicy(
        game,
        target_levels=exp4605._submitted_target_levels(),
        value_weight=exp4605._submitted_value_weight(),
    )
    prop = pol._proposer()
    t0 = time.time()
    ok = bool(prop._ensure_server())
    info = {
        "launch_ok": ok,
        "launch_s": round(time.time() - t0, 1),
        "pid": getattr(getattr(prop, "_proc", None), "pid", None),
        "repo_substr": getattr(prop, "repo_substr", None),
        "n_ctx": getattr(prop, "n_ctx", None),
        "port": getattr(prop, "port", None),
        "model_path": str(getattr(prop, "model_path", "")),
    }
    return (prop if ok else None), info


def stop_port(port: int) -> list[int]:
    """Stop whatever listens on `port` (by port ownership, never by name pattern)."""
    killed = []
    for line in _run(["ss", "-ltnpH"]).splitlines():
        if f"127.0.0.1:{port}" not in line:
            continue
        for tok in line.split("pid=")[1:]:
            digits = "".join(ch for ch in tok.split(",")[0] if ch.isdigit())
            if digits:
                try:
                    os.kill(int(digits), 15)
                    killed.append(int(digits))
                except OSError:
                    pass
    return killed


def install_shipped_proposer(exp4605) -> None:
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    def _policy_for_mode(mode: str, game: str):
        if mode == "bare":
            return E3AgentPolicy(
                game,
                proposer=exp4605._NoOpProposer(),
                target_levels=1,
                value_head=None,
                value_weight=0.0,
                candidate_router=None,
                navigation_cost_tiebreak=False,
            )
        pol = E3AgentPolicy(
            game,
            proposer=None,
            target_levels=exp4605._submitted_target_levels(),
            value_weight=exp4605._submitted_value_weight(),
        )
        _LOCAL.policy = pol
        return pol

    exp4605._policy_for_mode = _policy_for_mode


def run_cell(exp4605, spec: dict, budget: int) -> dict:
    cell = CELLS / f"llm_on__{spec['variant_signature'].replace('~', '_')}.json"
    if cell.exists():
        return json.loads(cell.read_text())
    _LOCAL.policy = None
    t0 = time.time()
    row, err = {}, ""
    try:
        row = dict(exp4605.run_variant_attempt("integrated", str(spec["game"]), spec, budget))
    except Exception as exc:  # a crashed cell stays visible
        err = repr(exc)[:400]
    pol = getattr(_LOCAL, "policy", None)
    witness = None
    if pol is not None:
        try:
            witness = pol.generator_liveness_witness()
        except Exception as exc:
            witness = {"witness_error": repr(exc)[:200]}
    out = {
        "arm": "llm_on",
        "game": str(spec["game"]),
        "variant": int(spec["variant"]),
        "variant_signature": spec["variant_signature"],
        "elapsed_s": round(time.time() - t0, 1),
        "cell_error": err,
        "first_win": bool(row.get("first_win")) if row else None,
        "reached_level": row.get("reached_level") if row else None,
        "actions": row.get("actions") if row else None,
        "row": row,
        "liveness_witness": witness,
    }
    CELLS.mkdir(parents=True, exist_ok=True)
    cell.write_text(json.dumps(out, indent=1, default=str))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True, help="CUDA index by PCI bus order")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--variants", default="1")
    ap.add_argument("--games", default="")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if not os.environ.get("CARNOT_LLAMA_SERVER"):
        print("BLOCKED: set CARNOT_LLAMA_SERVER to the CUDA llama-server binary", flush=True)
        return 2
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CARNOT_ARC_PROPOSER_PORT"] = str(args.port)
    os.environ["CARNOT_ARC_GATE_DEEPEN"] = "1"
    os.environ["CARNOT_ARC_GATE_VARIANT_IDS"] = args.variants
    os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
    os.environ.setdefault("CARNOT_ARC_SERVER_LOG_DIR", str(OUT / "server_logs"))
    Path(os.environ["CARNOT_ARC_SERVER_LOG_DIR"]).mkdir(parents=True, exist_ok=True)

    import carnot.experiment_4605_live_integration_scored_agent as exp4605

    games = (
        [g.strip() for g in args.games.split(",") if g.strip()]
        if args.games
        else exp4605._public_games(REPO)
    )
    vids = [int(v) for v in args.variants.replace(",", " ").split()]
    specs = exp4605.variant_specs(games, vids)
    meta = {
        "games": games,
        "variants": vids,
        "budget": args.budget,
        "k": args.k,
        "n_cells": len(specs),
        "git_head": _run(["git", "-C", str(REPO), "rev-parse", "HEAD"]),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": {
            k: os.environ.get(k)
            for k in (
                "CARNOT_LLAMA_SERVER",
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "CARNOT_ARC_PROPOSER_PORT",
                "CARNOT_ARC_SERVER_LOG_DIR",
            )
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    prop, srv = start_server(exp4605, games[0])
    meta["server"] = srv
    if prop is None:
        meta["blocked"] = "blocked_generator_server_launch_failed"
        (OUT / f"run{args.tag}.json").write_text(json.dumps(meta, indent=1, default=str))
        print("BLOCKED:", srv, flush=True)
        return 2
    meta["device"] = device_join(srv["pid"], args.gpu)
    print(f"[server] {srv} device={meta['device']['verdict']}", flush=True)
    if meta["device"]["verdict"] != "CONFIRMED_BY_PID_UUID_JOIN":
        print("BLOCKED: server not confirmed on the intended card", flush=True)
        stop_port(args.port)
        meta["blocked"] = "blocked_wrong_or_unverified_device"
        (OUT / f"run{args.tag}.json").write_text(json.dumps(meta, indent=1, default=str))
        return 2

    install_shipped_proposer(exp4605)
    t0 = time.time()
    rows = []
    try:
        with ThreadPoolExecutor(max_workers=args.k) as ex:
            futs = [ex.submit(run_cell, exp4605, s, args.budget) for s in specs]
            for f in as_completed(futs):
                rows.append(f.result())
                w = rows[-1].get("liveness_witness") or {}
                print(
                    f"[{len(rows)}/{len(specs)}] {rows[-1]['game']} "
                    f"attempts={w.get('induction_attempts_n')} "
                    f"planned={w.get('induction_attempts_planned')} "
                    f"wall={time.time() - t0:.0f}s",
                    flush=True,
                )
    finally:
        meta["wall_s"] = round(time.time() - t0, 1)
        meta["device_after"] = device_join(srv["pid"], args.gpu)
        meta["stopped_pids"] = stop_port(args.port)
        meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (OUT / f"run{args.tag}.json").write_text(json.dumps(meta, indent=1, default=str))
    return 0


def _health(port: int) -> bool:  # kept for manual checks
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return b"ok" in r.read()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
