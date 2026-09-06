#!/usr/bin/env python
"""Bounded 27B trial of the induction tool grammar (REQ-ARC-WMTE-7046), GPU 1 only.

WHAT THIS ANSWERS. The grammar transport (`CARNOT_ARC_INDUCE_TOOL_GRAMMAR=1`) had one
recorded trial, on the 0.5 GB CPU smoke model, and that trial's grammar admitted an
empty `arguments` object. With the grammar fixed, does the pinned Qwen3.8-27B produce
scoreable engines through the grammar loop? The control is the ordinary single-shot
induce path on the same model, same windows, same seed.

TWO ROLES, ONE FILE. `--role driver` owns the llama-server (launched by this script on
a non-default port, pinned to one card with CUDA_VISIBLE_DEVICES, killed by exact PID)
and runs one `--role cell` subprocess per (game, arm). A cell is its own process because
`E3_DIR` binds from `CARNOT_ARC_E3_DIR` at import time, so two arms in one process would
share an engine store and score each other's leftovers (the contamination the h2h
harness exp6440 documents).

RULES HONOURED. GPU placement is read from nvidia-smi joined on UUID, never from the env
var. A long-run receipt is armed (REQ-INFRA-6830). `results/**` is never written: the
engine store is redirected to the output directory. The server is stopped by PID.

FORCE TURN, AND WHY IT IS A FLAG (added 2026-09-06 for trial 2). Trial 1 hardcoded the
force-engine turn to 2, one turn earlier than the loop's own default of 3. Every grammar
cell then submitted only after the nudge, so whether the model submits unprompted was
never measured. `--force-turn` now carries that value, defaults to the loop's constant,
and is recorded in trial.json. Trial 1 is reproduced with `--force-turn 2`; it was not a
default and is not one now.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
LLAMA_SERVER = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
GGUF_NAME = "Qwen3.8-27B-Q4_K_M.gguf"
DEFAULT_GAMES = "tr87,sp80,sb26,re86,g50t"


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + msg, flush=True)


def _loop_default_force_turn() -> int:
    """The tool loop's own default force turn. Imported so it cannot drift from a copy."""
    sys.path.insert(0, str(REPO / "python"))
    from carnot.agentic.arc_induction_tool_loop import DEFAULT_FORCE_ENGINE_TURN

    return int(DEFAULT_FORCE_ENGINE_TURN)


def gpu_table() -> dict[str, int]:
    """{uuid: index} from nvidia-smi."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    table: dict[str, int] = {}
    for line in out.strip().splitlines():
        idx, uuid = [p.strip() for p in line.split(",")]
        table[uuid] = int(idx)
    return table


def gpu_apps() -> list[dict[str, Any]]:
    """Every compute process with the GPU INDEX it actually holds memory on."""
    table = gpu_table()
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, uuid, mem = [p.strip() for p in line.split(",")]
        rows.append({"pid": int(pid), "gpu_index": table.get(uuid), "used_mib": mem})
    return rows


def gpu_free_mib(index: int) -> int:
    out = subprocess.run(
        ["nvidia-smi", f"--id={index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return int(out.strip().splitlines()[0])


def orphan_xdist_workers() -> list[int]:
    """xdist workers hold GPU memory and are invisible to `pgrep -f pytest`."""
    found = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ")
        except OSError:
            continue
        if b"exec(eval(sys.stdin.readline()))" in cmd:
            found.append(int(p.name))
    return found


def http_json(url: str, payload: Optional[dict[str, Any]] = None, timeout: float = 10.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def health_ok(port: int) -> bool:
    try:
        return bool(http_json(f"http://127.0.0.1:{port}/health", timeout=2.0))
    except Exception:
        return False


def resolve_gguf() -> Optional[str]:
    root = Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots"
    for snap in sorted(root.glob("*")):
        cand = snap / GGUF_NAME
        if cand.exists():
            return str(cand)
    return None


# ----------------------------------------------------------------------------------
# cell
# ----------------------------------------------------------------------------------


def run_cell(args: argparse.Namespace) -> int:
    """One (game, arm). Env is already set by the driver; E3_DIR binds on import."""
    import hashlib

    import numpy as np

    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_induction_tool_loop as tool_loop_mod
    from carnot.agentic.arc_actions_to_progress import build_progress_window
    from carnot.agentic.arc_induction_tool_loop import induce_with_tool_loop
    from carnot.agentic.arc_induction_tools import InductionToolSession

    rec: dict[str, Any] = {
        "game": args.game,
        "arm": args.arm,
        "e3_dir": str(e3.E3_DIR),
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("CARNOT_ARC_") or k in ("CUDA_VISIBLE_DEVICES", "CARNOT_LLAMA_SERVER")
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t_cell = time.time()
    # Windows come from a pickle the driver prepared on CPU. Building one solves L1
    # by search and costs minutes; inside a GPU-timed cell that would burn the budget
    # while the server sits idle. The fallback builds only when no pickle is given.
    if args.window_pkl:
        import pickle

        with Path(args.window_pkl).open("rb") as fh:
            payload = pickle.load(fh)
        window, cell = list(payload["window"]), int(payload["cell"])
        rec["window_source"] = str(args.window_pkl)
    else:
        w = build_progress_window(args.game)
        if w is None:
            rec["status"] = "skipped_no_offline_window"
            out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
            return 0
        window, _full, cell = w
        window = list(window)
        rec["window_source"] = "build_progress_window"
    h = hashlib.sha256()
    for t in window:
        h.update(np.asarray(t.grid).tobytes())
        h.update(str(t.action).encode())
        h.update(np.asarray(t.next_grid).tobytes())
    rec["window_sha256"] = h.hexdigest()
    rec["window_rows"] = len(window)
    rec["cell"] = int(cell)

    p = e3.LocalGGUFProposer(
        port=args.port,
        mtp=False,
        kv_quant="q8_0",
        model_path=args.gguf,
        use_chat_template=True,
        tries=1,
    )
    rec["proposer"] = {
        "n_ctx": p.n_ctx,
        "max_tokens": p.max_tokens,
        "timeout": p.timeout,
        "repo_substr": p.repo_substr,
    }
    if not p._ensure_server():
        rec["status"] = "server_unavailable"
        out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
        return 2
    rec["server_reuse"] = {
        "model_check": getattr(p, "reuse_model_check", None),
        "n_ctx_check": getattr(p, "reuse_n_ctx_check", ""),
        "observed_model": getattr(p, "observed_server_model_path", None),
        "observed_n_ctx": getattr(p, "observed_server_n_ctx", None),
        "launched_own_server": bool(p.last_launch_argv),
        "port_after_ensure": p.port,
    }
    if p.last_launch_argv or p.port != args.port:
        # The cell must ride the driver's server. A relaunch means placement is unknown.
        rec["status"] = "refused_relaunched_server"
        try:
            if p._proc is not None:
                p._proc.terminate()
        except Exception:
            pass
        out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
        return 3

    wm = e3.E3_DIR / args.game / "world_model.py"
    try:
        wm.unlink()
    except FileNotFoundError:
        pass

    events: list[dict[str, Any]] = []
    t0 = time.time()
    if args.arm == "grammar":
        # WIRE-LEVEL GRAMMAR RECORD. The loop builds both grammars once per session from
        # a deep copy, so a mid-session change should be impossible. "Should be" is not a
        # measurement, and a grammar that changed under us would outrank everything else
        # this cell measures, so hash what each request actually carries.
        wire: list[dict[str, Any]] = []
        real_post = tool_loop_mod._post_chat

        def recording_post(*a: Any, **kw: Any) -> Any:
            g = kw.get("grammar")
            wire.append(
                {
                    "turn": kw.get("turn"),
                    "grammar_sha256": hashlib.sha256(g.encode()).hexdigest() if g else None,
                    "grammar_bytes": len(g.encode()) if g else 0,
                    "grammar_root": g.splitlines()[0] if g else None,
                }
            )
            return real_post(*a, **kw)

        tool_loop_mod._post_chat = recording_post
        try:
            ok, note = induce_with_tool_loop(
                p, args.game, window, int(cell), tool_event_sink=events
            )
        finally:
            tool_loop_mod._post_chat = real_post
        stats = dict(p.last_tool_loop_stats or {})
        rec["tool_loop_stats"] = stats
        rec["tool_events"] = events
        rec["grammar_wire_per_turn"] = wire
        distinct = sorted({w["grammar_sha256"] for w in wire if w["grammar_sha256"]})
        rec["grammar_distinct_sha256"] = distinct
        # THE MEASUREMENT. Which turn did the model first submit an engine on, and was
        # that before or at the turn the closure compels one? Read from the per-turn tool
        # names, not from a counter, so "unprompted" is not inferred.
        per_turn = stats.get("tool_calls_per_turn") or []
        rec["tool_names_per_turn"] = per_turn
        first = next(
            (i for i, names in enumerate(per_turn) if "run_engine_on_transitions" in (names or [])),
            None,
        )
        # Read the force turn from the loop's OWN resolver, not from this script's
        # argparse default: the cell is a subprocess and only the env var is authoritative.
        # The nudge fires at the END of turn index force-1, so the compelled request is
        # turn index `force`. Turn indices are 0-based throughout.
        force = tool_loop_mod._force_engine_turn()
        rec["first_submission_turn"] = first
        rec["force_engine_turn"] = force
        rec["first_submission_unprompted"] = None if first is None else first < force
        rec["tokens_decoded"] = stats.get("decode_tokens_total")
        rec["prompt_tokens_per_turn"] = stats.get("prompt_tokens_per_turn")
    else:
        ok, note = p.induce(args.game, window, int(cell))
        rec["tokens_decoded"] = getattr(p, "last_generated_tokens", None)
        rec["last_stop_type"] = getattr(p, "last_stop_type", "")
        rec["channel_totals"] = dict(getattr(p, "channel_totals", {}) or {})
        rec["raw_completion_chars"] = len(getattr(p, "last_raw_completion", "") or "")
        rec["reasoning_chars"] = len(getattr(p, "last_reasoning_content", "") or "")
    rec["wall_s"] = round(time.time() - t0, 2)
    rec["induce_ok"] = bool(ok)
    rec["induce_note"] = str(note)[:600]
    rec["engine_emitted"] = wm.exists()

    if wm.exists():
        code = wm.read_text()
        rec["engine_chars"] = len(code)
        rec["engine_sha256"] = hashlib.sha256(code.encode()).hexdigest()
        session = InductionToolSession(window, cell=int(cell))
        report = session.run_engine_on_transitions(code)
        rec["score"] = {
            "ok": bool(report.get("ok")),
            "error": report.get("error"),
            "accuracy": report.get("accuracy"),
            "cell_recall": report.get("cell_recall"),
            "held_out": report.get("held_out"),
            "n_engine_raised": report.get("n_engine_raised"),
            "static_defects": report.get("static_defects"),
            "is_memorizing": (report.get("memorization_scan") or {}).get("is_memorizing"),
        }
        try:
            engine, goal = e3.load_engine(args.game)
            rec["loadable"] = engine is not None and goal is not None
        except Exception as exc:  # noqa: BLE001
            rec["loadable"] = False
            rec["load_error"] = f"{type(exc).__name__}: {exc}"[:200]
    else:
        rec["score"] = {"ok": False, "error": "no world_model.py written"}
        rec["loadable"] = False
    rec["cell_wall_s"] = round(time.time() - t_cell, 2)
    rec["status"] = "done"
    out.write_text(json.dumps(rec, indent=2, sort_keys=True, default=str) + "\n")
    return 0


# ----------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------


def run_driver(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO / "python"))
    from carnot.testing.long_run_receipt import install

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cells").mkdir(exist_ok=True)
    (out / "relaunch_logs").mkdir(exist_ok=True)
    trial: dict[str, Any] = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": vars(args),
        "cells": [],
    }
    install(out / "receipt.json", progress=lambda: {"cells_done": len(trial["cells"])})

    def save() -> None:
        (out / "trial.json").write_text(
            json.dumps(trial, indent=2, sort_keys=True, default=str) + "\n"
        )

    # ---- preconditions (Pre-Launch Preconditions Discipline) ----
    gguf = resolve_gguf()
    pre = [
        {"resource": "gguf_cached_qwen38_27b", "available": gguf is not None, "detail": gguf},
        {
            "resource": "llama_server_cuda_binary",
            "available": LLAMA_SERVER.exists(),
            "detail": str(LLAMA_SERVER),
        },
    ]
    table = gpu_table()
    idx_to_uuid = {v: k for k, v in table.items()}
    pre.append(
        {
            "resource": f"gpu{args.gpu}_present",
            "available": args.gpu in idx_to_uuid,
            "detail": idx_to_uuid.get(args.gpu),
        }
    )
    free = gpu_free_mib(args.gpu) if args.gpu in idx_to_uuid else 0
    pre.append(
        {"resource": f"gpu{args.gpu}_free_mib>=21000", "available": free >= 21000, "detail": free}
    )
    pre.append(
        {"resource": f"port_{args.port}_free", "available": not health_ok(args.port), "detail": ""}
    )
    orphans = orphan_xdist_workers()
    pre.append({"resource": "no_orphan_xdist_workers", "available": not orphans, "detail": orphans})
    trial["preconditions_checked"] = pre
    save()
    for c in pre:
        log(f"precondition {c['resource']}: {'ok' if c['available'] else 'MISSING'} {c['detail']}")
    if not all(c["available"] for c in pre if c["resource"] != "no_orphan_xdist_workers"):
        trial["status"] = "blocked_preconditions"
        save()
        return 2

    # ---- launch the server on ONE card ----
    server_args = [
        str(LLAMA_SERVER),
        "-m",
        str(gguf),
        "-ngl",
        "999",
        "-c",
        str(args.n_ctx),
        "--port",
        str(args.port),
        "--host",
        "127.0.0.1",
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--parallel",
        "1",
        "-fit",
        "off",
        "--jinja",
        "--reasoning-format",
        "deepseek",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    srv_log = out / "llama_server.log"
    log(f"launch: CUDA_VISIBLE_DEVICES={args.gpu} {' '.join(server_args)}")
    proc = subprocess.Popen(
        server_args, stdout=srv_log.open("ab"), stderr=subprocess.STDOUT, env=env
    )
    trial["server"] = {
        "pid": proc.pid,
        "argv": server_args,
        "cuda_visible_devices": str(args.gpu),
        "log": str(srv_log),
    }
    save()
    t_launch = time.time()
    rc = 0
    try:
        while time.time() - t_launch < 900:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early rc={proc.returncode}; see {srv_log}")
            if health_ok(args.port):
                break
            time.sleep(2)
        else:
            raise RuntimeError("llama-server never became healthy within 900 s")
        t_healthy = time.time()
        trial["server"]["health_wait_s"] = round(t_healthy - t_launch, 1)
        placement = [r for r in gpu_apps() if r["pid"] == proc.pid]
        trial["server"]["placement_measured"] = placement
        props = http_json(f"http://127.0.0.1:{args.port}/props", timeout=20)
        trial["server"]["props_model"] = str(
            (
                props.get("model_path")
                or props.get("default_generation_settings", {}).get("model")
                or props
            )
        )[:300]
        trial["server"]["props_n_ctx"] = (props.get("default_generation_settings") or {}).get(
            "n_ctx"
        )
        save()
        log(f"healthy in {trial['server']['health_wait_s']}s; placement={placement}")
        cards = sorted({r["gpu_index"] for r in placement})
        if cards != [args.gpu]:
            raise RuntimeError(
                f"server pid {proc.pid} holds memory on GPUs {cards}, not only {args.gpu}; stopping"
            )
        if GGUF_NAME not in json.dumps(props):
            raise RuntimeError(f"/props does not name {GGUF_NAME}")

        # ---- cells ----
        games = [g.strip() for g in args.games.split(",") if g.strip()]
        base_env = dict(
            os.environ,
            PYTHONPATH=str(REPO / "python"),
            JAX_PLATFORMS="cpu",
            CUDA_VISIBLE_DEVICES=str(args.gpu),
            CARNOT_LLAMA_SERVER=str(LLAMA_SERVER),
            CARNOT_ARC_SERVER_LOG_DIR=str(out / "relaunch_logs"),
            CARNOT_ARC_LLM_BACKEND="llamacpp",
            CARNOT_ARC_GENERATOR_SEED=str(args.seed),
            CARNOT_ARC_INDUCE_N_CTX=str(args.n_ctx),
            CARNOT_ARC_INDUCE_MAX_TOKENS=str(args.max_tokens),
            CARNOT_ARC_INDUCE_TIMEOUT=str(args.timeout),
            CARNOT_ARC_MTP="0",
            CARNOT_ARC_FFN_CPU_LAYERS="0",
            CARNOT_ARC_LLAMA_SERVER_PARALLEL="1",
        )
        for k in (
            "CARNOT_ARC_INDUCE_TOOL_LOOP",
            "CARNOT_ARC_INDUCE_TOOL_GRAMMAR",
            "CARNOT_ARC_INDUCE_TOOL_COMPACT",
            "CARNOT_ARC_INDUCE_TOOL_SELFPARSE",
        ):
            base_env.pop(k, None)
        arm_env = {
            "grammar": {
                "CARNOT_ARC_INDUCE_TOOL_LOOP": "1",
                "CARNOT_ARC_INDUCE_TOOL_GRAMMAR": "1",
                "CARNOT_ARC_INDUCE_TOOL_TURNS": str(args.turns),
                "CARNOT_ARC_INDUCE_TOOL_FORCE_ENGINE_TURN": str(args.force_turn),
            },
            "control": {
                "CARNOT_ARC_INDUCE_THINK": "1",
                "CARNOT_ARC_INDUCE_THINKING_BUDGET": str(args.think_budget),
            },
        }
        windows_dir = Path(args.windows_dir) if args.windows_dir else None
        for game in games:
            window_pkl = (windows_dir / f"{game}.pkl") if windows_dir else None
            if window_pkl is not None and not window_pkl.exists():
                trial["cells"].append({"game": game, "status": "skipped_no_window_pickle"})
                save()
                log(f"no window pickle for {game}; skipping both arms")
                continue
            for arm in ("control", "grammar"):
                elapsed_min = (time.time() - t_healthy) / 60.0
                if elapsed_min > args.budget_min:
                    trial["cells"].append(
                        {
                            "game": game,
                            "arm": arm,
                            "status": "not_started_budget_exhausted",
                            "elapsed_min": round(elapsed_min, 1),
                        }
                    )
                    save()
                    log(f"budget exhausted at {elapsed_min:.1f} min; skipping {game}/{arm}")
                    continue
                cell_out = out / "cells" / f"{game}_{arm}.json"
                cenv = dict(base_env, **arm_env[arm], CARNOT_ARC_E3_DIR=str(out / "e3" / arm))
                cmd = [
                    sys.executable,
                    str(HERE),
                    "--role",
                    "cell",
                    "--game",
                    game,
                    "--arm",
                    arm,
                    "--port",
                    str(args.port),
                    "--gguf",
                    str(gguf),
                    "--out",
                    str(cell_out),
                ]
                if window_pkl is not None:
                    cmd += ["--window-pkl", str(window_pkl)]
                log(f"cell {game}/{arm} start (elapsed {elapsed_min:.1f} min)")
                c0 = time.time()
                clog = (out / "cells" / f"{game}_{arm}.log").open("ab")
                try:
                    cp = subprocess.run(
                        cmd,
                        env=cenv,
                        stdout=clog,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout + 240,
                    )
                    crc: Any = cp.returncode
                except subprocess.TimeoutExpired:
                    crc = "timeout"
                row: dict[str, Any] = {
                    "game": game,
                    "arm": arm,
                    "rc": crc,
                    "driver_wall_s": round(time.time() - c0, 1),
                }
                if cell_out.exists():
                    try:
                        row["record"] = json.loads(cell_out.read_text())
                    except json.JSONDecodeError:
                        row["record_error"] = "unreadable"
                trial["cells"].append(row)
                save()
                r = row.get("record", {})
                log(
                    f"cell {game}/{arm} done rc={crc} wall={row['driver_wall_s']}s "
                    f"emitted={r.get('engine_emitted')} scoreable={(r.get('score') or {}).get('ok')} "
                    f"acc={(r.get('score') or {}).get('accuracy')} tokens={r.get('tokens_decoded')} "
                    f"term={(r.get('tool_loop_stats') or {}).get('terminated_by')}"
                )
        trial["status"] = "complete"
    except Exception as exc:  # noqa: BLE001
        trial["status"] = f"aborted: {type(exc).__name__}: {exc}"[:400]
        log(trial["status"])
        rc = 1
    finally:
        # ---- stop the server by EXACT pid, then verify ----
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.kill(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        trial["server"]["exit_code"] = proc.returncode
        time.sleep(3)
        trial["after"] = {
            "port_listening": health_ok(args.port),
            "gpu_apps": gpu_apps(),
            "gpu_free_mib": {i: gpu_free_mib(i) for i in sorted(idx_to_uuid)},
            "orphan_xdist_workers": orphan_xdist_workers(),
        }
        trial["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        save()
        log(f"server pid {proc.pid} stopped rc={proc.returncode}; after={trial['after']}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", choices=["driver", "cell"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8939)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--n-ctx", type=int, default=49152)
    ap.add_argument("--max-tokens", type=int, default=6144)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--turns", type=int, default=4)
    # The turn at which the loop stops asking for a submission and constrains one.
    # Trial 1 (2026-09-05) passed 2, which fired the nudge one turn earlier than the
    # loop's own default and left unprompted submission unmeasured. The default here
    # is imported from the loop, never retyped, so "the loop default" stays true if
    # that constant moves.
    ap.add_argument("--force-turn", type=int, default=_loop_default_force_turn())
    ap.add_argument("--think-budget", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--budget-min", type=float, default=38.0)
    ap.add_argument("--games", default=DEFAULT_GAMES)
    ap.add_argument("--game")
    ap.add_argument("--arm", choices=["control", "grammar"])
    ap.add_argument("--gguf")
    ap.add_argument("--windows-dir", help="driver: directory of <game>.pkl windows")
    ap.add_argument("--window-pkl", help="cell: the window pickle for --game")
    args = ap.parse_args()
    if args.role == "cell":
        return run_cell(args)
    return run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
