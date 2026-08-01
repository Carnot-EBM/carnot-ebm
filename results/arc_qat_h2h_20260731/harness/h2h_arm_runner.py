#!/usr/bin/env python
"""QAT vs non-QAT head-to-head for the ARC live inducer -- gemma-4-31B-it.

DERIVED FROM `results/inducer_h2h_6021/h2h_arm_runner.py.frozen`, 2026-07-31, by changing the
ARMS dict and NOTHING ELSE. N_CTX (32768), BUDGET (16384), KV_QUANT (q8_0), GPU pinning, the
per-cell wedge guards, and ROSTER/TRIALS (imported from exp5760, 13 games x 3 trials) are all
byte-identical to the run that justified adopting gemma-4-31B. That is the point: the only
variable that moves is the quantisation, so any separation is attributable to it.

WHY THIS RUN EXISTS. The 11-0-2 / p=0.00098 result that selected gemma-4-31B was measured on
Q4_K_M. `unsloth/gemma-4-31B-it-qat-GGUF` UD-Q4_K_XL is quantisation-aware-trained and 1.0 GB
smaller (17.3 vs 18.3 GB), and its card claims near-bf16 quality -- but that is a model-card
claim with no perplexity or accuracy numbers behind it. Adopting it on that basis would be an
unverified substitution. This measures it.
"""

# ruff: noqa: E402, E501, N806
# Deliberate, and inherited from the frozen exp6021 runner this is derived from:
# E402 late imports are load-bearing (arc_executable_world_model.E3_DIR binds at IMPORT
# time, so the arm's CARNOT_ARC_E3_DIR must be exported before the import runs), and
# E501/N806 are the original's style. Reformatting the measurement harness after it
# produced the result would break the byte-provenance claim the artifact makes.

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
sys.path.insert(0, str(REPO / "python"))

# PARALLEL ARMS (2026-07-31). Originally both arms ran on GPU 1 sequentially. They now run
# CONCURRENTLY on one card each, halving wall-clock ~10.3h -> ~5.2h. This is a deliberate
# departure from the frozen protocol and it is recorded as such: the run is no longer
# byte-identical to exp6021, only equivalent-in-substance.
#
# Why it is defensible HERE: the measured outcome is induction QUALITY (held-out accuracy),
# not throughput, and the two cards are physically identical RTX 3090s. A quality metric does
# not care which of two identical cards produced it. It would NOT be defensible for a
# tok/s comparison, where card identity and thermal state are the measurement.
#
# The per-card residency proof is UNCHANGED in strength -- each arm still proves which card it
# actually landed on by UUID, from residency rather than from the env var. Only the expected
# UUID is now per-arm instead of a single module constant.
GPU_INDEX = 1  # default / legacy; each ARM overrides via "gpu_index"
GPU1_UUID = "GPU-7971baff-9583-eaa6-2292-393f930a28f9"
GPU0_UUID = "GPU-b52387a2-c625-de87-8d34-e6f64e684bab"
N_CTX = 32768  # exp5764's successfully-deployed n_ctx_deployed. NOT 81920 (inflates the footprint).
BUDGET = 16384  # exp5726/5760/5764 completion budget
MIN_RESIDENCY_MIB = 15000  # a real Q4 27B/31B offload; below this the model is not on the card
KV_QUANT = "q8_0"

ARMS: dict[str, dict[str, Any]] = {
    # CONTROL: the currently-shipped quantisation. This is the arm whose 11-0-2 result
    # justified adopting gemma-4-31B at all, so it is the only legitimate baseline.
    "q4km": {
        "label": "gemma-4-31B-it Q4_K_M (non-QAT, shipped)",
        # Distinct substrings on purpose. Both arms are the SAME model family, so a substring
        # matching both would silently defeat the identity guard -- and `_resolve_gguf` globs
        # `models--*<substr>*GGUF` over cache DIRECTORY names, where "gemma-4-31B-it" matches
        # both `...-it-GGUF` and `...-it-qat-GGUF`. `model_path` below is explicit, so
        # resolution never depends on the glob, but the substring is still kept discriminating.
        "repo_substr": "gemma-4-31B-it-GGUF",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "gguf": "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/snapshots/f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf",
        "port": 8981,
        "quant": "Q4_K_M",
        "gpu_index": 1,
        "gpu_uuid": GPU1_UUID,
    },
    # TREATMENT: quantisation-aware training, and 1.0 GB smaller.
    "qat": {
        "label": "gemma-4-31B-it QAT UD-Q4_K_XL",
        "repo_substr": "gemma-4-31B-it-qat-GGUF",
        "hf_id": "unsloth/gemma-4-31B-it-qat-GGUF",
        "gguf": "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-qat-GGUF/snapshots/43cc1aeb31adf47ec06a854507ce552cd9862e6f/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf",
        "port": 8982,
        "quant": "UD-Q4_K_XL",
        "gpu_index": 0,
        "gpu_uuid": GPU0_UUID,
    },
}


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------------------------------
# GPU / liveness probes
# --------------------------------------------------------------------------------------
def gpu1_present(gpu_index: int, gpu_uuid: str) -> bool:
    """Is THIS ARM's GPU still on the PCI bus? The eGPU has fallen off mid-run before
    (operator power-cycle required), so this is checked per cell, not once at launch."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=25,
        )
        return r.returncode == 0 and gpu_uuid in r.stdout
    except Exception:
        return False


def pid_residency_mib(pid: int, gpu_uuid: str) -> int | None:
    """MiB our server PID holds ON THIS ARM's UUID. This is how we PROVE which card we got --
    CUDA_VISIBLE_DEVICES is an intention; residency is a fact. With two arms running at once
    the proof matters MORE, not less: a mis-set env var would now land both servers on one
    card, and the residency check is what would catch it."""
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
        if len(p) == 3 and p[0].isdigit() and int(p[0]) == pid and p[1] == gpu_uuid:
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
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(arm["gpu_index"]))
    log(f"  launch: CUDA_VISIBLE_DEVICES={arm['gpu_index']} {' '.join(args)}")
    t0 = time.time()
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
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
    resident = pid_residency_mib(proc.pid, arm["gpu_uuid"])
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
        "gpu_uuid_proven": arm["gpu_uuid"],
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


def terminate(proc: subprocess.Popen | None) -> None:
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
    from carnot.experiment_5760_cegis_refinement_induction_ab import (  # noqa: E402
        ROSTER as ROSTER_13,
    )
    from carnot.experiment_5760_cegis_refinement_induction_ab import (
        TRIALS,
    )

    # 20-GAME EXTENSION (2026-07-31). The 13-game run returned a null (5-2-6, p=0.453) with
    # only 7 discordant pairs, because 5 of the 6 ties were BOTH-ZERO floor games where
    # neither arm induced anything. The sign test is over GAMES, so the lever is more games.
    #
    # These 7 are ALL the remaining public games that have a usable offline L1 window -- not
    # a subset chosen for effect. The other 5 non-roster games (dc22, ka59, sc25, tn36, wa30)
    # are excluded because `build_progress_window` cannot build one, a property of the
    # fixtures that is independent of either arm's score. See PREREGISTRATION_20games.md,
    # committed BEFORE these cells ran.
    ROSTER_EXTRA_7 = ["bp35", "cn04", "lf52", "ls20", "m0r0", "s5i5", "su15"]
    ROSTER = list(ROSTER_13) + ROSTER_EXTRA_7
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
    add("gpu_on_pci_bus", gpu1_present(arm["gpu_index"], arm["gpu_uuid"]), arm["gpu_uuid"])
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
    wedge: dict[str, Any] | None = None
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
            if not gpu1_present(arm["gpu_index"], arm["gpu_uuid"]):
                wedge = {"kind": "gpu1_lost_mid_run", "game": game, "trial": t}
                log(f"WEDGE {wedge}")
                break
            res = pid_residency_mib(proc.pid, arm["gpu_uuid"])
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
