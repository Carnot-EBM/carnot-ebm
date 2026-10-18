#!/usr/bin/env python
"""ONE ARM of the equalized-budget REAL-holdout induction A/B (exp6474).

WHAT THIS MEASURES. Whether single-shot induction or the tool loop
(REQ-ARC-WMTE-6460) produces world models that GENERALIZE, on a holdout that is
actually out of sample. The analysis that motivated this found the exp5726/6440
lineage's `heldout_accuracy` is IN-SAMPLE fit: the induce prompt has shown ALL
window transitions since 2026-08-01, and memorizing engines score HIGHER on it
(0.857 vs 0.684 mean, n=30). No genuine generalization number exists for the
single-shot path. This experiment measures one, identically for both arms.

THE THREE EQUALIZATIONS (the design, verbatim from the brief):
  1. Same visible evidence. Both arms see `holdout_split(window)[0]` -- the same
     function the tool session uses, imported, not re-implemented. Single-shot
     is prompted with ONLY those rows; the tool loop's session computes the
     identical split from the same window.
  2. Same holdout, big enough to mean something: the held-out window tail PLUS
     every same-level transition of `full_traj` outside the window. This turns
     a 2-3-row holdout into tens of rows for most games.
  3. Same total decode ceiling: 49,152 tokens per cell for both arms.
       single-shot: max_tokens=49152, tries=1  (1 x 49,152)
       tool loop:   max_tokens=4096 x 12 turns (12 x 4,096 = 49,152)
     WHY 49k AND NOT 102k: a plain think-on Qwen3.8 induce measures ~41.4k
     tokens (exp6440), so 49,152 does not starve single-shot -- and it is the
     loop's SHIPPED ceiling, so the loop arm is the deployment being decided
     on. tries=1 because tries=3 would triple single-shot's ceiling; the
     loop's internal iteration is the design difference under test, not a
     retry inequity.

WALL-TIME IS NOT REPORTED as a comparison: the two arms run concurrently on
sibling cards sharing CPU/memory bandwidth, which invalidated a wall gate
earlier today. This run measures QUALITY.

ARM ISOLATION. One arm per process (E3_DIR binds at import), one card per arm.
The stale-engine trap: a FAILED induce leaves the previous cell's engine on
disk, so the store file is deleted before every induce and scoring is gated on
this cell's own induce_ok (the exp5722 attribution bug, guarded here).
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

MIN_RESIDENCY_MIB = 15000
SEED_BASE = 100  # exp6440's paired-seed convention: seed = base + trial, same both arms
TOTAL_DECODE_CEILING = 49152
TRIALS = [0, 1, 2]


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
    """MiB the server PID holds on the named card. An env var is an intention;
    residency is the fact."""
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
    """Bounded REAL /completion: /health can answer 200 while /completion hangs."""
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
            json.load(r)
        return {"alive": True, "s": round(time.time() - t0, 2)}
    except Exception as exc:
        return {
            "alive": False,
            "s": round(time.time() - t0, 2),
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }


def server_binary_linkage(server: str) -> dict[str, Any]:
    try:
        r = subprocess.run(["ldd", server], capture_output=True, text=True, timeout=25)
        return {
            "libcuda": "libcuda.so.1" in r.stdout,
            "libcublas": "libcublas" in r.stdout,
            "libamdhip64": "libamdhip64" in r.stdout,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:160]}


def resolve_gguf() -> str:
    root = Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots"
    hits = sorted(root.glob("*/Qwen3.8-27B-Q4_K_M.gguf")) if root.exists() else []
    if not hits:
        raise SystemExit(f"Qwen3.8-27B-Q4_K_M.gguf not found under {root}")
    return str(hits[-1])


def transport_env(arm: str, tool_transport: str) -> dict[str, str]:
    """Env the chosen transport needs, as a pure function so it can be tested off-GPU.

    The single arm NEVER gets the variable: setting it would silently turn the control arm
    into a second treatment arm, and the whole A/B would compare a loop against a loop.
    """

    if arm != "tool" or tool_transport != "selfparse":
        return {}
    return {"CARNOT_ARC_INDUCE_TOOL_LOOP": "selfparse"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["single", "tool"])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--gpu", required=True, type=int, help="physical CUDA index for this arm")
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("--games", help="comma-separated subset of the roster")
    # TRANSPORT, added 2026-08-29. The tool arm previously had exactly one transport --
    # native `tools` in the request -- because that was the only one that existed. The
    # selfparse transport (REQ-ARC-WMTE-6730) sends no `tools` field and parses the
    # model's XML agent-side, which is what the SCORED vLLM server requires: it is
    # launched with no --enable-auto-tool-choice and returns HTTP 400 on any request
    # carrying `tools`. Default stays "native" so previously banked cells keep meaning
    # what they meant; a selfparse run belongs in its own run-dir, never mixed in.
    ap.add_argument(
        "--tool-transport",
        choices=["native", "selfparse"],
        default="native",
        help="tool arm only: 'native' sends a tools field (dev-only), "
        "'selfparse' carries schemas as prompt text and parses XML agent-side",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    e3_dir = run_dir / f"e3_{args.arm}"
    e3_dir.mkdir(exist_ok=True)
    shard = run_dir / f"shard_{args.arm}.jsonl"
    metap = run_dir / f"meta_{args.arm}.json"

    # ---- env BEFORE any carnot import ----
    os.environ["CARNOT_ARC_E3_DIR"] = str(e3_dir)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CARNOT_ARC_SERVER_LOG_DIR"] = str(run_dir / "server_logs")
    os.environ["CARNOT_ARC_MTP"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = str(args.gpu)
    # The single arm must take the SHIPPED single-shot path; the tool arm calls
    # the loop directly, and compaction stays at its shipped default (unset) --
    # compaction is a separate lever, not what this A/B measures.
    # The single arm must never see this set. For the tool arm the value selects the
    # TRANSPORT: popping it (native) makes the loop send a `tools` field, which only the
    # local llama.cpp dev twin accepts.
    os.environ.pop("CARNOT_ARC_INDUCE_TOOL_LOOP", None)
    for key, value in transport_env(args.arm, args.tool_transport).items():
        os.environ[key] = value
    os.environ.pop("CARNOT_ARC_INDUCE_TOOL_COMPACT", None)
    if not os.environ.get("CARNOT_LLAMA_SERVER"):
        log("FATAL: CARNOT_LLAMA_SERVER unset (a module relaunch could pick the HIP build)")
        return 3

    from carnot.agentic import arc_actions_to_progress as atp  # noqa: E402
    from carnot.agentic.arc_executable_world_model import (  # noqa: E402
        E3_DIR,
        LocalGGUFProposer,
        WorldModelVerifier,
    )
    from carnot.agentic.arc_induction_tool_loop import induce_with_tool_loop  # noqa: E402
    from carnot.agentic.arc_induction_tools import (  # noqa: E402
        _exec_candidate,
        holdout_split,
        memorization_scan,
        window_changed_coords,
    )
    from carnot.experiment_5760_cegis_refinement_induction_ab import (  # noqa: E402
        ROSTER as _ROSTER,
    )

    if str(E3_DIR) != str(e3_dir):
        log(f"FATAL: CARNOT_ARC_E3_DIR not honoured (E3_DIR={E3_DIR})")
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
    gpu_uuid = gpu_uuid_for_index(args.gpu)
    linkage = server_binary_linkage(os.environ["CARNOT_LLAMA_SERVER"])

    precond: list[dict[str, Any]] = []

    def add(res: str, ok: bool, detail: str = "") -> None:
        precond.append({"resource": res, "available": bool(ok), "detail": str(detail)[:200]})

    add("gguf_cached_qwen38_27b_q4km", Path(gguf).exists(), gguf)
    add(
        "llama_server_binary",
        Path(os.environ["CARNOT_LLAMA_SERVER"]).exists(),
        os.environ["CARNOT_LLAMA_SERVER"],
    )
    add(
        "server_binary_is_cuda_build",
        bool(linkage.get("libcuda")) and not linkage.get("libamdhip64"),
        json.dumps(linkage),
    )
    add(f"gpu{args.gpu}_on_pci_bus", bool(gpu_uuid), gpu_uuid)
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

    # ---- windows + shared splits + extended holdout ----
    log(f"building {len(roster)} windows...")
    cells_def: dict[str, dict[str, Any]] = {}
    for g in roster:
        w = atp.build_progress_window(g)
        if w is None:
            log(f"  SKIP {g}: no offline L1 window")
            continue
        window, full_traj, cell = list(w[0]), list(w[1]), int(w[2])
        visible, held_tail = holdout_split(window)
        window_ids = {id(t) for t in window}
        lvl = getattr(window[0], "level_before", 0)
        # Same-level rows of the full trajectory the window (and so the prompt)
        # never contained. Genuinely out of sample for BOTH arms.
        extra = [
            t
            for t in full_traj
            if id(t) not in window_ids
            and getattr(t, "level_before", 0) == lvl
            and getattr(t, "level_after", 0) == lvl
        ]
        holdout_rows = list(held_tail) + extra
        cells_def[g] = {
            "window": window,
            "visible": visible,
            "held_tail": held_tail,
            "holdout_rows": holdout_rows,
            "cell": cell,
            "n_visible": len(visible),
            "n_holdout": len(holdout_rows),
            "n_extra_from_traj": len(extra),
        }
        log(
            f"  {g}: window={len(window)} visible={len(visible)} "
            f"holdout={len(holdout_rows)} (tail {len(held_tail)} + traj {len(extra)})"
        )

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
    pending = [(g, t) for g in roster if g in cells_def for t in TRIALS if (g, t) not in done]
    total = sum(1 for g in roster if g in cells_def for _ in TRIALS)
    log(f"arm={args.arm} resume: {len(done)}/{total} cells cached; {len(pending)} pending")

    prop = LocalGGUFProposer(
        repo_substr="Qwen3.8-27B",
        model_path=gguf,
        port=args.port,
        mtp=False,
        kv_quant="q8_0",
        max_tokens=(TOTAL_DECODE_CEILING if args.arm == "single" else 4096),
        timeout=3600,
    )
    if args.arm == "single":
        # tries=1: the total-decode equalization. tries=3 would give single-shot a
        # 3x49k ceiling against the loop's 12x4096.
        prop.tries = 1

    meta: dict[str, Any] = {
        "arm": args.arm,
        "gguf": gguf,
        "hf_id": "unsloth/Qwen3.8-27B-GGUF",
        "quantisation": "Q4_K_M",
        "port_requested": args.port,
        "gpu_index": args.gpu,
        "gpu_uuid": gpu_uuid,
        "n_ctx": prop.n_ctx,
        "decode_ceiling_total": TOTAL_DECODE_CEILING,
        "decode_shape": ("1 x 49152, tries=1" if args.arm == "single" else "12 turns x 4096"),
        # Recorded so a shard can never be read as the other transport's evidence.
        "tool_transport": (args.tool_transport if args.arm == "tool" else None),
        "induce_timeout_s": 3600,
        "kv_quant": "q8_0",
        "seed_base": SEED_BASE,
        "trials": TRIALS,
        "roster": [g for g in roster if g in cells_def],
        "holdout_sizes": {g: cells_def[g]["n_holdout"] for g in cells_def},
        "visible_sizes": {g: cells_def[g]["n_visible"] for g in cells_def},
        "e3_dir": str(e3_dir),
        "server_binary": os.environ["CARNOT_LLAMA_SERVER"],
        "server_binary_linkage": linkage,
        "preconditions_checked": precond,
        "cells_total": total,
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if not pending:
        meta["status"] = "complete_cached"
        meta["cells_completed"] = len(done)
        metap.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return 0

    t_arm = time.time()
    if not prop._ensure_server():
        meta["status"] = "blocked"
        meta["honest_verdict"] = "blocked_generator_server_unavailable"
        metap.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        log("BLOCKED: _ensure_server failed")
        return 2
    pid = prop._proc.pid if prop._proc is not None else -1
    res0 = pid_residency_mib(pid, gpu_uuid)
    props = props_model_path(prop.port)
    live = completion_alive(prop.port)
    meta["server"] = {
        "pid": pid,
        "port_actual": prop.port,
        "residency_mib": res0,
        "props_model_path": props,
        "completion_probe": live,
        "stderr_log": str(getattr(prop, "_stderr_log_path", "")),
        "launch_argv": list(getattr(prop, "last_launch_argv", ())),
    }
    log(f"server pid={pid} port={prop.port} residency_gpu{args.gpu}={res0} MiB live={live}")
    if res0 is None or res0 < MIN_RESIDENCY_MIB:
        meta["status"] = "blocked"
        meta["honest_verdict"] = "blocked_no_gpu_residency"
        metap.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return 2
    if Path(gguf).name not in props or not live["alive"]:
        meta["status"] = "blocked"
        meta["honest_verdict"] = "blocked_server_identity_or_liveness"
        metap.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return 2

    def score_rows(src: str, rows: list) -> dict[str, Any]:
        """Identical scorer for both arms: exec the stored module, score with the
        guarded WorldModelVerifier. Level-up rows are excluded by the verifier."""
        engine, err = _exec_candidate(src, "engine")
        if engine is None:
            return {"error": err}
        try:
            vr = WorldModelVerifier(list(rows)).score(engine)
        except Exception as exc:  # noqa: BLE001
            return {"error": f"verifier raised {type(exc).__name__}: {exc}"[:200]}
        return {
            "n_gradeable": int(vr.n),
            "n_correct": int(vr.n_correct),
            "accuracy": round(float(vr.accuracy), 4),
            "cell_recall": round(float(vr.cell_recall), 4),
        }

    wedge: Optional[dict[str, Any]] = None
    for game, trial in pending:
        res = pid_residency_mib(pid, gpu_uuid)
        if res is None or res < MIN_RESIDENCY_MIB:
            wedge = {"kind": "residency_collapsed", "game": game, "residency_mib": res}
            log(f"WEDGE {wedge}")
            break
        lv = completion_alive(prop.port)
        if not lv["alive"]:
            wedge = {"kind": "health_ok_completion_hung", "game": game, "probe": lv}
            log(f"WEDGE {wedge}")
            break
        d = cells_def[game]
        os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(SEED_BASE + trial)
        # Stale-engine guard (exp5722): a failed induce must score as a failure,
        # never as the previous cell's engine.
        store = e3_dir / game / "world_model.py"
        store.unlink(missing_ok=True)
        log(f"RUN {args.arm} {game} trial={trial} (residency={res} MiB)")
        c0 = time.time()
        try:
            if args.arm == "single":
                ok, note = prop.induce(game, list(d["visible"]), int(d["cell"]))
            else:
                ok, note = induce_with_tool_loop(prop, game, list(d["window"]), int(d["cell"]))
        except Exception as exc:  # noqa: BLE001
            ok, note = False, f"cell_crash: {type(exc).__name__}: {exc}"[:300]
        elapsed = round(time.time() - c0, 2)
        src = store.read_text() if (ok and store.exists()) else ""
        row: dict[str, Any] = {
            "game": game,
            "trial": trial,
            "arm": args.arm,
            "induce_ok": bool(ok) and bool(src),
            "note": str(note)[:300],
            "elapsed_s": elapsed,
            "seed": os.environ["CARNOT_ARC_GENERATOR_SEED"],
            "residency_mib_at_cell_start": res,
            "n_visible": d["n_visible"],
            "n_holdout": d["n_holdout"],
        }
        if args.arm == "tool":
            row["tool_loop_stats"] = copy.deepcopy(getattr(prop, "last_tool_loop_stats", {}) or {})
        else:
            row["last_stop_type"] = getattr(prop, "last_stop_type", "")
        if row["induce_ok"]:
            row["holdout"] = score_rows(src, d["holdout_rows"])
            row["visible_fit"] = score_rows(src, d["visible"])
            scan = memorization_scan(src, window_changed_coords(d["visible"]))
            row["is_memorizing"] = bool(scan.get("is_memorizing"))
            row["memorization_scan"] = scan
        with shard.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        done[(game, trial)] = row
        h = (row.get("holdout") or {}).get("accuracy")
        log(
            f"  -> induce_ok={row['induce_ok']} holdout={h} "
            f"visible={(row.get('visible_fit') or {}).get('accuracy')} "
            f"mem={row.get('is_memorizing')} {elapsed}s"
        )

    try:
        prop._terminate_stale_proc("arm complete; stopping the arm's own server")
    except Exception as exc:  # noqa: BLE001
        log(f"terminate warn: {type(exc).__name__}: {exc}")

    meta["cells_completed"] = len(done)
    meta["cells_missing"] = [
        [g, t] for g in roster if g in cells_def for t in TRIALS if (g, t) not in done
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
