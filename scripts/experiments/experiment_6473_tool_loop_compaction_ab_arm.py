#!/usr/bin/env python
"""ONE ARM of the compacted-carried-state pilot A/B (exp6473, REQ-ARC-WMTE-6540 Section 11).

WHAT THIS MEASURES. Phase 0 + Phase 1 of the design note
`docs/research-notes/arc-induction-compacted-carried-state-2026-08-19.md`:
  * Phase 0: the OFF arm's measured `prompt_tokens_per_turn` replaces the note's
    transcript-growth ESTIMATES. Gate: if end-of-loop context rarely exceeds ~25k,
    STOP before Phase 1.
  * Phase 1: 13 paired cells (the exp5760 13-game roster x trial 0), compaction OFF
    vs ON, same seeds, same windows, tool loop enabled in both arms.

ARM ISOLATION. One arm per process, because `arc_executable_world_model.E3_DIR` binds
at import time from `CARNOT_ARC_E3_DIR` -- the same contamination reason exp6440 runs
one arm per subprocess. This runner sets the env BEFORE any carnot import.

CELL MECHANISM. `induce_with_tool_loop(proposer, game, window, cell)` is called
DIRECTLY (the loop the design measures), not through `proposer.induce()`. Deliberate:
`induce()` falls back to single-shot when the loop fails, which would spend up to a
second full timeout per failing cell and mix single-shot decode into the loop's
numbers. A loop failure is recorded as its honest `terminated_by` instead.

HARDWARE DISCIPLINE (operator brief 2026-08-20). GPU 0 ONLY -- GPU 1 runs an
unrelated experiment. The parent shell must export CARNOT_LLAMA_SERVER (the CUDA
build; priority-1 override so a relaunch cannot pick the HIP/iGPU build),
CUDA_VISIBLE_DEVICES=0 (priority-1 inherits ambient env, so THIS is what pins the
card), CARNOT_ARC_GENERATOR_CUDA_GPU=0, and this runner exports
CARNOT_ARC_SERVER_LOG_DIR into the run dir. Residency is verified per cell from
nvidia-smi -- an env var is an intention; residency is a fact.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

MIN_RESIDENCY_MIB = 15000  # below this the 27B Q4_K_M did not really load on the card
SEED_BASE_DEFAULT = 100  # exp6440's paired-seed convention


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def gpu_uuid_for_index(idx: int) -> str:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=25,
    )
    for line in r.stdout.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 2 and p[0] == str(idx):
            return p[1]
    return ""


def pid_residency_mib(pid: int, uuid: str) -> Optional[int]:
    """MiB the server PID holds on the named card. Proves WHICH card we got."""
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=25,
        )
    except Exception:
        return None
    total = 0
    for line in r.stdout.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 3 and p[0].isdigit() and int(p[0]) == pid and p[1] == uuid:
            total += int(p[2])
    return total or None


def health_ok(port: int, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return b"ok" in r.read()
    except Exception:
        return False


def props_model_path(port: int, timeout: float = 20.0) -> str:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=timeout) as r:
            d = json.loads(r.read())
        return str(
            d.get("model_path") or d.get("default_generation_settings", {}).get("model") or ""
        )
    except Exception as exc:
        return f"PROPS_ERROR {type(exc).__name__}: {exc}"


def completion_alive(port: int, timeout: float = 120.0) -> dict[str, Any]:
    """A bounded REAL /completion. /health can answer 200 while /completion hangs
    (the exp5833 failure), so health alone is never trusted as liveness."""
    body = json.dumps(
        {"prompt": "2+2=", "n_predict": 8, "temperature": 0.0, "stream": False}
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
        return {
            "alive": True,
            "s": round(time.time() - t0, 2),
            "content": str(d.get("content"))[:60],
        }
    except Exception as exc:
        return {
            "alive": False,
            "s": round(time.time() - t0, 2),
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }


def server_binary_linkage(server: str) -> dict[str, Any]:
    """ldd evidence that the binary is the CUDA build, not HIP. A plausible duration
    is not proof of the right device; this plus residency is the proof."""
    try:
        r = subprocess.run(["ldd", server], capture_output=True, text=True, timeout=25)
        out = r.stdout
        return {
            "libcuda": "libcuda.so.1" in out,
            "libcublas": "libcublas" in out,
            "libamdhip64": "libamdhip64" in out,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:160]}


def resolve_gguf() -> str:
    """Qwen3.8-27B Q4_K_M from the HF cache. Raises on a miss; never substitutes."""
    root = Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots"
    hits = sorted(root.glob("*/Qwen3.8-27B-Q4_K_M.gguf")) if root.exists() else []
    if not hits:
        raise SystemExit(f"Qwen3.8-27B-Q4_K_M.gguf not found under {root}")
    return str(hits[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["off", "on"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--port", type=int, default=8983)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--seed-base", type=int, default=SEED_BASE_DEFAULT)
    ap.add_argument("--games", help="comma-separated subset of the roster (default: all 13)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    e3_dir = run_dir / f"e3_{args.arm}"
    e3_dir.mkdir(exist_ok=True)
    shard = run_dir / f"shard_{args.arm}.jsonl"
    metap = run_dir / f"meta_{args.arm}.json"

    # ---- env BEFORE any carnot import (E3_DIR binds at import time) ----
    os.environ["CARNOT_ARC_E3_DIR"] = str(e3_dir)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CARNOT_ARC_INDUCE_TOOL_LOOP"] = "1"
    if args.arm == "on":
        os.environ["CARNOT_ARC_INDUCE_TOOL_COMPACT"] = "1"  # defaults: growth 8192, budget 2048
    else:
        os.environ.pop("CARNOT_ARC_INDUCE_TOOL_COMPACT", None)
    os.environ["CARNOT_ARC_SERVER_LOG_DIR"] = str(run_dir / "server_logs")
    os.environ["CARNOT_ARC_MTP"] = "0"  # the live default; no MTP head for Qwen3.8

    # Hardware discipline: refuse to run half-configured rather than silently
    # measuring the wrong device (the recent iGPU-migration incident).
    required_env = {
        "CARNOT_LLAMA_SERVER": os.environ.get("CARNOT_LLAMA_SERVER", ""),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "CARNOT_ARC_GENERATOR_CUDA_GPU": os.environ.get("CARNOT_ARC_GENERATOR_CUDA_GPU", ""),
    }
    missing_env = [k for k, v in required_env.items() if not v]
    if missing_env:
        log(f"FATAL: required env unset: {missing_env}")
        return 3
    if required_env["CUDA_VISIBLE_DEVICES"] != "0":
        log("FATAL: CUDA_VISIBLE_DEVICES must be '0' -- GPU 1 runs an unrelated experiment")
        return 3

    from carnot.agentic import arc_actions_to_progress as atp  # noqa: E402
    from carnot.agentic.arc_executable_world_model import (  # noqa: E402
        E3_DIR,
        LocalGGUFProposer,
    )
    from carnot.agentic.arc_induction_tool_loop import induce_with_tool_loop  # noqa: E402
    from carnot.experiment_5760_cegis_refinement_induction_ab import (  # noqa: E402
        ROSTER as _ROSTER,
    )

    if str(E3_DIR) != str(e3_dir):
        log(f"FATAL: CARNOT_ARC_E3_DIR not honoured (E3_DIR={E3_DIR}) -- contamination risk")
        return 3

    roster = list(_ROSTER)
    if args.games:
        want = [g.strip() for g in args.games.split(",") if g.strip()]
        unknown = [g for g in want if g not in roster]
        if unknown:
            log(f"FATAL: --games names {unknown} not in roster {roster}")
            return 3
        roster = want

    gguf = resolve_gguf()
    gpu0_uuid = gpu_uuid_for_index(0)

    # ---- preconditions (Pre-Launch Preconditions Discipline) ----
    precond: list[dict[str, Any]] = []

    def add(res: str, ok: bool, detail: str = "") -> None:
        precond.append({"resource": res, "available": bool(ok), "detail": str(detail)[:200]})

    linkage = server_binary_linkage(required_env["CARNOT_LLAMA_SERVER"])
    add("gguf_cached_qwen38_27b_q4km", Path(gguf).exists(), gguf)
    add(
        "llama_server_binary",
        Path(required_env["CARNOT_LLAMA_SERVER"]).exists(),
        required_env["CARNOT_LLAMA_SERVER"],
    )
    add(
        "server_binary_is_cuda_build",
        bool(linkage.get("libcuda")) and not linkage.get("libamdhip64"),
        json.dumps(linkage),
    )
    add("gpu0_on_pci_bus", bool(gpu0_uuid), gpu0_uuid)
    add(f"port_{args.port}_free", not health_ok(args.port), "no pre-existing server on our port")
    if not all(c["available"] for c in precond):
        missing = [c["resource"] for c in precond if not c["available"]]
        metap.write_text(
            json.dumps(
                {
                    "arm": args.arm,
                    "status": "blocked",
                    "honest_verdict": f"blocked_{'_'.join(missing)[:90]}",
                    "preconditions_checked": precond,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        log(f"BLOCKED: {missing}")
        return 2

    # ---- windows (identical corpus both arms: same builder, same fixtures) ----
    log(f"building {len(roster)} windows...")
    windows: dict[str, Any] = {}
    for g in roster:
        w = atp.build_progress_window(g)
        windows[g] = w
        if w is None:
            log(f"  SKIP {g}: no offline L1 window")

    # ---- resume ----
    done: dict[tuple[str, int], dict[str, Any]] = {}
    if shard.exists():
        for line in shard.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done[(r["game"], int(r["trial"]))] = r
            except (json.JSONDecodeError, KeyError):
                continue
    pending = [
        (g, args.trial)
        for g in roster
        if windows.get(g) is not None and (g, args.trial) not in done
    ]
    total = sum(1 for g in roster if windows.get(g) is not None)
    log(f"arm={args.arm} resume: {len(done)}/{total} cells cached; {len(pending)} pending")

    prop = LocalGGUFProposer(
        repo_substr="Qwen3.8-27B",
        model_path=gguf,
        port=args.port,
        mtp=False,
        kv_quant="q8_0",
        max_tokens=4096,  # the local live default (CARNOT_ARC_INDUCE_MAX_TOKENS unset)
        timeout=2400,  # the scored per-call timeout the design note cites
    )

    meta: dict[str, Any] = {
        "arm": args.arm,
        "compaction_env": os.environ.get("CARNOT_ARC_INDUCE_TOOL_COMPACT", "<unset>"),
        "gguf": gguf,
        "hf_id": "unsloth/Qwen3.8-27B-GGUF",
        "quantisation": "Q4_K_M",
        "port_requested": args.port,
        "n_ctx": prop.n_ctx,
        "max_tokens_per_turn": 4096,
        "induce_timeout_s": 2400,
        "kv_quant": "q8_0",
        "seed_base": args.seed_base,
        "trial": args.trial,
        "roster": roster,
        "e3_dir": str(e3_dir),
        "server_binary": required_env["CARNOT_LLAMA_SERVER"],
        "server_binary_linkage": linkage,
        "cuda_visible_devices": required_env["CUDA_VISIBLE_DEVICES"],
        "gpu0_uuid": gpu0_uuid,
        "preconditions_checked": precond,
        "cells_total": total,
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if not pending:
        meta["status"] = "complete_cached"
        meta["cells_completed"] = len(done)
        metap.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return 0

    # ---- server up + identity/residency proof BEFORE any cell ----
    t_arm = time.time()
    if not prop._ensure_server():
        meta["status"] = "blocked"
        meta["honest_verdict"] = "blocked_generator_server_unavailable"
        metap.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        log("BLOCKED: _ensure_server failed")
        return 2
    pid = prop._proc.pid if prop._proc is not None else -1
    res0 = pid_residency_mib(pid, gpu0_uuid)
    props = props_model_path(prop.port)
    live = completion_alive(prop.port)
    meta["server"] = {
        "pid": pid,
        "port_actual": prop.port,
        "residency_mib_gpu0": res0,
        "props_model_path": props,
        "completion_probe": live,
        "stderr_log": str(getattr(prop, "_stderr_log_path", "")),
        "launch_argv": list(getattr(prop, "last_launch_argv", ())),
        "generator_server_path": str(getattr(prop, "generator_server_path", "")),
    }
    log(f"server pid={pid} port={prop.port} residency_gpu0={res0} MiB live={live.get('alive')}")
    if res0 is None or res0 < MIN_RESIDENCY_MIB:
        log(f"FATAL: no real GPU-0 offload (residency={res0} MiB)")
        meta["status"] = "blocked"
        meta["honest_verdict"] = "blocked_no_gpu0_residency"
        metap.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return 2
    if Path(gguf).name not in props:
        log(f"FATAL: /props identity mismatch: {props!r}")
        meta["status"] = "blocked"
        meta["honest_verdict"] = "blocked_props_model_mismatch"
        metap.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return 2
    if not live["alive"]:
        log(f"FATAL: /health ok but /completion dead: {live}")
        meta["status"] = "blocked"
        meta["honest_verdict"] = "blocked_completion_probe_dead"
        metap.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return 2

    wedge: Optional[dict[str, Any]] = None
    for game, trial in pending:
        # Per-cell wedge checks: a wedge must be a recorded fact, not a hang.
        res = pid_residency_mib(pid, gpu0_uuid)
        if res is None or res < MIN_RESIDENCY_MIB:
            wedge = {"kind": "residency_collapsed", "game": game, "residency_mib": res}
            log(f"WEDGE {wedge}")
            break
        pp = props_model_path(prop.port)
        if Path(gguf).name not in pp:
            wedge = {"kind": "props_identity_lost", "game": game, "props": pp[:200]}
            log(f"WEDGE {wedge}")
            break
        lv = completion_alive(prop.port)
        if not lv["alive"]:
            wedge = {"kind": "health_ok_completion_hung", "game": game, "probe": lv}
            log(f"WEDGE {wedge}")
            break

        # Paired seed: identical per (game, trial) across the two arms, so the
        # sampling randomness is differenced out (exp6440's pairing rationale).
        os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(args.seed_base + trial)
        w = windows[game]
        log(f"RUN {args.arm} {game} trial={trial} (residency={res} MiB)")
        c0 = time.time()
        try:
            ok, note = induce_with_tool_loop(prop, game, list(w[0]), int(w[2]))
        except Exception as exc:  # noqa: BLE001 - a cell crash is a row, not an arm loss
            ok, note = False, f"cell_crash: {type(exc).__name__}: {exc}"[:300]
        stats = copy.deepcopy(getattr(prop, "last_tool_loop_stats", {}) or {})
        row = {
            "game": game,
            "trial": trial,
            "arm": args.arm,
            "ok": bool(ok),
            "note": str(note)[:300],
            "elapsed_s": round(time.time() - c0, 2),
            "seed": os.environ["CARNOT_ARC_GENERATOR_SEED"],
            "residency_mib_at_cell_start": res,
            "stats": stats,
        }
        with shard.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        done[(game, trial)] = row
        pts = [p for p in stats.get("prompt_tokens_per_turn", []) if isinstance(p, (int, float))]
        log(
            f"  -> ok={ok} terminated_by={stats.get('terminated_by')} turns={stats.get('turns')} "
            f"prompt_max={max(pts) if pts else None} compactions={stats.get('compactions')} "
            f"parse_fail={stats.get('tool_call_parse_failures')} "
            f"holdout={stats.get('best_holdout_accuracy')} {row['elapsed_s']}s"
        )

    try:
        prop._terminate_stale_proc("arm complete; stopping the arm's own server")
    except Exception as exc:  # noqa: BLE001
        log(f"terminate warn: {type(exc).__name__}: {exc}")

    meta["cells_completed"] = len(done)
    meta["cells_missing"] = [
        [g, args.trial]
        for g in roster
        if windows.get(g) is not None and (g, args.trial) not in done
    ]
    meta["wedge"] = wedge
    meta["arm_wall_s"] = round(time.time() - t_arm, 2)
    meta["status"] = "complete" if not meta["cells_missing"] else "partial"
    meta["ended_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    metap.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    log(f"ARM DONE status={meta['status']} completed={len(done)}/{total} wedge={wedge}")
    return 0 if meta["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
