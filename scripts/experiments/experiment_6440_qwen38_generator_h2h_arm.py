#!/usr/bin/env python
"""ONE ARM of the inducer head-to-head, now three-way: Qwen3.6-27B vs gemma-4-31B-it vs Qwen3.8-27B.

REQ-ARC-GEN-6440 (operator 2026-08-14: "we should compare Gemma-4-31b to the newly released
Qwen-3.8-27b"). Qwen3.8-27B is a NEW model (dense 27B, 262K context, MTP-trained, released August
2026), not the Qwen3.6-27B this harness retired on 2026-07-28 -- so the Failed-Experiment Rerun
Discipline is satisfied by a real forward difference, not a relabel.

PROVENANCE: this file is a VERBATIM COPY of
`results/arc_gemma31b_migration_evidence_20260728/h2h_arm_runner.py` with one arm added. It is a
copy rather than an edit because `results/**` is EVIDENCE -- read it, never write it. Copying also
keeps the retired comparison runnable exactly as it was.

WHY THE PROTOCOL IS NOT TOUCHED. Same 13 games, 3 trials, Q4_K_M both sides, n_ctx 32768, budget
16384, q8_0 KV, per-arm engine store. Changing any of those would make the new number
incomparable with the 11-0-2 result it is meant to be read against, and an incomparable
measurement is the failure this project spent 2026-08-13 correcting.

WHY GEMMA IS RE-RUN RATHER THAN READ FROM THE 2026-07-28 SHARD. The archived shard is 17 days old.
Between then and now the venv, llama.cpp build and driver may all have moved. Scoring a fresh arm
against a stale comparator would attribute infrastructure drift to the model. The gemma re-run
doubles as a reproducibility check on the original 38/39.

ORIGINAL DOCSTRING FOLLOWS.
--------------------------------------------------------------------------------------------------
ONE ARM of the inducer head-to-head: base Qwen3.6-27B vs gemma-4-31B-it on world-model induction.

WHY a separate process per arm (verbose per CLAUDE.md): `arc_executable_world_model.E3_DIR` is bound
AT IMPORT TIME from `CARNOT_ARC_E3_DIR`. A single process therefore cannot give two arms separate
engine stores, and a shared `results/arc_e3/<game>/world_model.py` is exactly how an earlier run in
this project got contaminated (arm B scoring arm A's leftover engine) and had to be discarded. So
each arm runs as its own subprocess with its own `CARNOT_ARC_E3_DIR` exported BEFORE python starts.

MECHANISM IDENTITY: the cell is `exp5726.run_reason_cell_budget` VERBATIM at budget=16384, which is
byte-identical to what exp5764 ran for the gemma arm. Corpus/repeats come from exp5760's ROSTER and
TRIALS -- the same 13 games and 3 trials, imported not retyped, so they cannot drift.

WEDGE SURVIVABILITY: three prior attempts at this measurement died on infrastructure. So, per cell,
BEFORE running it we re-verify (a) GPU 1 still exists on the PCI bus, (b) our server PID still holds
>15GB of residency ON GPU 1's UUID -- proving which card we actually got from residency, never from
the env var, (c) /props still reports OUR gguf (the default-port trap: port 8919 is held by an AMD
iGPU 9B server, so a mis-pointed proposer would silently measure the wrong model), and (d) a bounded
REAL /completion succeeds. (d) is the important one: exp5833 died with /health returning 200 while
/completion HUNG, so a health check is NOT a liveness check.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
sys.path.insert(0, str(REPO / "python"))

GPU_INDEX = 1  # both arms run here, sequentially. GPU 0 is another lane's (verified per-PID).
GPU1_UUID = "GPU-7971baff-9583-eaa6-2292-393f930a28f9"
N_CTX = 32768  # exp5764's successfully-deployed n_ctx_deployed. NOT 81920 (inflates the footprint).
BUDGET = 16384  # exp5726/5760/5764 completion budget
MIN_RESIDENCY_MIB = 15000  # a real Q4 27B/31B offload; below this the model is not on the card
KV_QUANT = "q8_0"

ARMS: dict[str, dict[str, Any]] = {
    "qwen27b": {
        "label": "qwen3.6-27B-base",
        "repo_substr": "Qwen3.6-27B",
        "hf_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
        "gguf": (
            "/home/ianblenke/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-MTP-GGUF/"
            "snapshots/5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q4_K_M.gguf"
        ),
        "port": 8977,  # explicit: NEVER the 8919 default (held by an iGPU 9B server)
        "quant": "Q4_K_M",
    },
    "gemma31b": {
        "label": "gemma-4-31B-it",
        "repo_substr": "gemma-4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "gguf": (
            "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/"
            "snapshots/f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf"
        ),
        "port": 8978,
        "quant": "Q4_K_M",
    },
    # ADDED 2026-08-14. Q4_K_M to match both incumbents exactly -- a UD-Q4_K_XL arm would be a
    # different quantisation and its result would not be readable against the 11-0-2 tally.
    # Port is explicit and distinct for the same reason the other two are: 8919 is held by an AMD
    # iGPU 9B server, and a mis-pointed proposer silently measures the wrong model.
    "qwen38_27b": {
        "label": "qwen3.8-27B",
        "repo_substr": "Qwen3.8-27B",
        "hf_id": "unsloth/Qwen3.8-27B-GGUF",
        "gguf": None,  # resolved from the HF cache at launch; see _resolve_gguf
        "port": 8979,
        "quant": "Q4_K_M",
    },
}


def _resolve_gguf(arm: dict) -> str:
    """Resolve a GGUF path from the HF cache instead of hardcoding a snapshot hash.

    The two incumbent arms carry literal snapshot paths, which is fine for a frozen re-run and
    wrong for a model downloaded today -- the hash is not knowable when this file is written. The
    lookup is exact on the repo folder and the filename, and it RAISES on a miss rather than
    falling back to any .gguf it can find: silently measuring a different model is the failure the
    explicit ports in this file already guard against.
    """
    if arm.get("gguf"):
        return str(arm["gguf"])
    repo = arm["hf_id"].replace("/", "--")
    root = Path.home() / ".cache/huggingface/hub" / f"models--{repo}" / "snapshots"
    want = f"{arm['repo_substr']}-{arm['quant']}.gguf"
    hits = sorted(root.glob(f"*/{want}")) if root.exists() else []
    if not hits:
        raise SystemExit(
            f"{arm['label']}: {want} not found under {root}. Download it before running this arm; "
            "this runner will not substitute a different quantisation."
        )
    return str(hits[-1])


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------------------
# GPU / liveness probes
# --------------------------------------------------------------------------------------
def gpu1_present() -> bool:
    """Is GPU 1 still on the PCI bus? The eGPU has fallen off mid-run before (operator
    power-cycle required), so this is checked per cell, not once at launch."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "-i", str(GPU_INDEX), "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=25,
        )
        return r.returncode == 0 and GPU1_UUID in r.stdout
    except Exception:
        return False


def pid_residency_mib(pid: int) -> Optional[int]:
    """MiB our server PID holds ON GPU 1's UUID. This is how we PROVE which card we got --
    CUDA_VISIBLE_DEVICES is an intention; residency is a fact."""
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
    for line in r.stdout.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 3 and p[0].isdigit() and int(p[0]) == pid and p[1] == GPU1_UUID:
            return int(p[2])
    return None


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
    """A BOUNDED REAL /completion. Distinguishes the exp5833 failure mode -- /health 200 while
    /completion hangs -- from a genuinely live server. A urllib timeout here IS the wedge signal."""
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
            "content": str(d.get("content"))[:80],
        }
    except Exception as exc:
        return {
            "alive": False,
            "s": round(time.time() - t0, 2),
            "error": f"{type(exc).__name__}: {exc}"[:200],
        }


def health_ok(port: int, timeout: float = 3.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as r:
            return b"ok" in r.read()
    except Exception:
        return False


# --------------------------------------------------------------------------------------
# Server lifecycle. Teardown is BY EXPLICIT PID -- never a pkill pattern (which on this box
# would match this very command line).
# --------------------------------------------------------------------------------------
def launch(arm: dict[str, Any], llama_server: Path) -> tuple[subprocess.Popen, dict[str, Any]]:
    args = [
        str(llama_server),
        "-m",
        arm["gguf"],
        "-ngl",
        "999",
        "-c",
        str(N_CTX),
        "--port",
        str(arm["port"]),
        "--host",
        "127.0.0.1",
        "--cache-type-k",
        KV_QUANT,
        "--cache-type-v",
        KV_QUANT,
        "-fit",
        "off",
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(GPU_INDEX))
    log(f"  launch: CUDA_VISIBLE_DEVICES={GPU_INDEX} {' '.join(args)}")
    t0 = time.time()
    # CAPTURE the server's stderr instead of discarding it (changed 2026-08-15). The original
    # sent both streams to DEVNULL. On the first Qwen3.8 run the server died mid-arm and the
    # harness recorded only `Connection refused` -- a crash with no log is undiagnosable, so the
    # run could not be classified as a model problem or an infrastructure one, and re-running
    # blind would have reproduced exactly the same non-answer.
    srv_log = (
        Path(os.environ.get("CARNOT_H2H_SERVER_LOG_DIR", "/tmp"))
        / f"llama_server_{args[args.index('--port') + 1]}.log"
    )
    srv_fh = srv_log.open("ab")
    log(f"  server stderr -> {srv_log}")
    proc = subprocess.Popen(args, stdout=srv_fh, stderr=subprocess.STDOUT, env=env)
    deadline = t0 + 1800
    healthy = False
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early rc={proc.returncode}")
        if health_ok(arm["port"]):
            healthy = True
            break
        time.sleep(2)
    if not healthy:
        terminate(proc)
        raise RuntimeError("llama-server never became healthy within 1800s")
    health_s = round(time.time() - t0, 1)
    resident = pid_residency_mib(proc.pid)
    props = props_model_path(arm["port"])
    live = completion_alive(arm["port"])
    meta = {
        "pid": proc.pid,
        "port": arm["port"],
        "n_ctx_deployed": N_CTX,
        "health_wait_s": health_s,
        "residency_mib_gpu1": resident,
        "props_model_path": props,
        "completion_probe": live,
        "gpu_uuid_proven": GPU1_UUID,
    }
    log(f"  healthy in {health_s}s; residency={resident} MiB on GPU1; live={live.get('alive')}")
    # GATES: all three must hold or we STOP rather than measure something unknown.
    if resident is None or resident < MIN_RESIDENCY_MIB:
        terminate(proc)
        raise RuntimeError(f"no real GPU-1 offload: residency={resident} MiB < {MIN_RESIDENCY_MIB}")
    if Path(arm["gguf"]).name not in props:
        terminate(proc)
        raise RuntimeError(
            f"/props model mismatch: {props!r} does not contain {Path(arm['gguf']).name}"
        )
    if not live["alive"]:
        terminate(proc)
        raise RuntimeError(f"health 200 but /completion dead: {live}")
    return proc, meta


def terminate(proc: Optional[subprocess.Popen]) -> None:
    if proc is None:
        return
    try:
        pid = proc.pid
        proc.terminate()
        try:
            proc.wait(timeout=45)
        except Exception:
            log(f"  SIGKILL explicit pid {pid}")
            proc.kill()
            proc.wait(timeout=45)
    except Exception as exc:
        log(f"  terminate warn: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--shard", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()
    arm = ARMS[args.arm]
    # Resolve once, here, so every downstream use (server launch, /props identity check, the
    # recorded artifact) reads the SAME path. Resolving at each use site is how two of them end up
    # pointing at different snapshots.
    arm["gguf"] = _resolve_gguf(arm)
    shard = Path(args.shard)
    metap = Path(args.meta)

    # E3_DIR must already be redirected by the PARENT via env before this process started.
    from carnot.agentic.arc_executable_world_model import E3_DIR  # noqa: E402

    want_e3 = os.environ.get("CARNOT_ARC_E3_DIR", "")
    log(f"arm={args.arm} E3_DIR={E3_DIR} (env={want_e3})")
    if not want_e3 or str(E3_DIR) != want_e3:
        log("FATAL: CARNOT_ARC_E3_DIR not honoured -- refusing to run (contamination risk)")
        return 3

    from carnot.agentic import arc_actions_to_progress as atp  # noqa: E402
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer  # noqa: E402
    from carnot.experiment_5726_thinkingcap_16k_dualgpu_reason_ab import (  # noqa: E402
        LLAMA_SERVER,
        run_reason_cell_budget,
    )
    from carnot.experiment_5760_cegis_refinement_induction_ab import ROSTER, TRIALS  # noqa: E402
    from carnot.experiment_5764_gemma31b_singleshot_induction_ab import (  # noqa: E402
        _window_changed_coords,
        memorization_scan,
    )

    # ---- Preconditions (Pre-Launch Preconditions Discipline) BEFORE any inference ----
    precond: list[dict[str, Any]] = []

    def add(res: str, ok: bool, detail: str = "") -> None:
        precond.append({"resource": res, "available": bool(ok), "detail": detail})

    add("gguf_cached::" + arm["label"], Path(arm["gguf"]).exists(), arm["gguf"])
    add("llama_server_binary", Path(LLAMA_SERVER).exists(), str(LLAMA_SERVER))
    add("gpu1_on_pci_bus", gpu1_present(), GPU1_UUID)
    try:
        from llama_cpp import llama_cpp as _b

        add("llama_cpp_gpu_offload", bool(_b.llama_supports_gpu_offload()), "CUDA build check")
    except Exception as exc:
        add("llama_cpp_gpu_offload", False, f"{type(exc).__name__}: {exc}"[:160])
    # The default-port trap: 8919 is held by an AMD-iGPU 9B server. Our port must be OURS.
    add(
        f"port_{arm['port']}_free_or_ours",
        not health_ok(arm["port"]),
        "no pre-existing server on our port",
    )

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

    # ---- Windows (identical corpus for both arms; built from the same offline fixtures) ----
    log(f"building {len(ROSTER)} windows...")
    windows: dict[str, Any] = {}
    for g in ROSTER:
        w = atp.build_progress_window(g)
        windows[g] = w
        if w is None:
            log(f"  SKIP {g}: no offline L1 window")

    # ---- Resume: per-cell cache, so a wedge costs at most ONE cell ----
    done: dict[tuple[str, int], dict[str, Any]] = {}
    if shard.exists():
        for line in shard.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[(r["game"], int(r["trial"]))] = r
    pending = [
        (g, t) for g in ROSTER for t in TRIALS if windows.get(g) is not None and (g, t) not in done
    ]
    total = sum(1 for g in ROSTER for _ in TRIALS if windows.get(g) is not None)
    log(f"resume: {len(done)}/{total} cells cached; {len(pending)} pending")

    meta: dict[str, Any] = {
        "arm": args.arm,
        "label": arm["label"],
        "gguf": arm["gguf"],
        "quantisation": arm["quant"],
        "hf_id": arm["hf_id"],
        "n_ctx_deployed": N_CTX,
        "budget": BUDGET,
        "kv_quant": KV_QUANT,
        "use_chat_template": True,
        "mtp": False,
        "roster": list(ROSTER),
        "trials": list(TRIALS),
        "e3_dir": str(E3_DIR),
        "preconditions_checked": precond,
        "cells_total": total,
        "started_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    if not pending:
        meta["status"] = "complete_cached"
        meta["cells_completed"] = len(done)
        metap.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        return 0

    proc = None
    wedge: Optional[dict[str, Any]] = None
    t_arm = time.time()
    try:
        proc, srv = launch(arm, Path(LLAMA_SERVER))
        meta["server"] = srv
        prop = LocalGGUFProposer(
            repo_substr=arm["repo_substr"],
            port=arm["port"],
            mtp=False,
            kv_quant=KV_QUANT,
            n_ctx=N_CTX,
            max_tokens=BUDGET,
            timeout=1800,
            use_chat_template=True,
            model_path=arm["gguf"],
        )
        for game, t in pending:
            # ---- per-cell guards: a wedge must be a RECORDED FACT, not a hang ----
            if not gpu1_present():
                wedge = {"kind": "gpu1_lost_mid_run", "game": game, "trial": t}
                log(f"WEDGE {wedge}")
                break
            res = pid_residency_mib(proc.pid)
            if res is None or res < MIN_RESIDENCY_MIB:
                wedge = {
                    "kind": "residency_collapsed",
                    "game": game,
                    "trial": t,
                    "residency_mib": res,
                }
                log(f"WEDGE {wedge}")
                break
            pp = props_model_path(arm["port"])
            if Path(arm["gguf"]).name not in pp:
                wedge = {"kind": "props_identity_lost", "game": game, "trial": t, "props": pp[:200]}
                log(f"WEDGE {wedge}")
                break
            lv = completion_alive(arm["port"])
            if not lv["alive"]:
                wedge = {
                    "kind": "health_ok_completion_hung",
                    "game": game,
                    "trial": t,
                    "probe": lv,
                }
                log(f"WEDGE {wedge}")
                break

            log(f"RUN {args.arm} {game} trial={t} (residency={res} MiB, probe={lv['s']}s)")
            c0 = time.time()
            try:
                row = run_reason_cell_budget(
                    game,
                    prop,
                    trial=t,
                    window=windows[game][0],
                    full_traj=windows[game][1],
                    cell=windows[game][2],
                    budget=BUDGET,
                )
                try:
                    src = (E3_DIR / game / "world_model.py").read_text()
                except Exception:
                    src = ""
                ms = memorization_scan(src, _window_changed_coords(windows[game][0]))
                row["mem_scan"] = ms
                row["is_memorizing"] = ms["is_memorizing"]
            except Exception as exc:
                row = {
                    "error": f"cell_crash: {type(exc).__name__}: {exc}"[:300],
                    "heldout_accuracy": None,
                }
            row["game"] = game
            row["trial"] = t
            row["arm"] = args.arm
            row["generator"] = arm["label"]
            row["elapsed_s"] = round(time.time() - c0, 2)
            row["residency_mib_at_cell_start"] = res
            row["server_n_ctx"] = N_CTX
            with shard.open("a") as f:  # PER-CELL cache write
                f.write(json.dumps(row) + "\n")
            done[(game, t)] = row
            log(
                f"  -> induce_ok={row.get('induce_ok')} heldout={row.get('heldout_accuracy')} "
                f"cell_recall={row.get('cell_recall')} reason={row.get('reason_engaged')} "
                f"overran={row.get('overran')} mem={row.get('is_memorizing')} "
                f"{row['elapsed_s']}s"
            )
    except Exception as exc:
        wedge = {"kind": "arm_exception", "error": f"{type(exc).__name__}: {exc}"[:400]}
        log(f"ARM EXCEPTION: {wedge}")
    finally:
        terminate(proc)

    meta["cells_completed"] = len(done)
    meta["cells_missing"] = [
        [g, t] for g in ROSTER for t in TRIALS if windows.get(g) is not None and (g, t) not in done
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
