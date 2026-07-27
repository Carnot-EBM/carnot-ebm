#!/usr/bin/env python
"""Re-measure the exp4605 held-out first-win rate with a WORKING generator at
eval-representative concurrency.

WHY THIS EXISTS
---------------
The baseline the whole ARC programme is steered by --
`first_win_rate_integrated = 0.04`, ops/known-issues.md:3484 -- was produced by
`python/carnot/experiment_4605_live_integration_scored_agent.py`, whose BOTH arms
install `_NoOpProposer` (line 722), i.e. the number was measured with the LLM tier
switched OFF and at concurrency 1. The 2026-07-27 generator concurrency fault
degrades the agent to exactly that LLM-OFF shape while it still reports itself as the
LLM-on scored path. So the operator's question ("did the fault depress our measured
first-win rate?") cannot be answered by re-reading 4605: it has to be re-measured with
the LLM tier ON, at a concurrency >= 2 (the threshold the fault actually fires at).

DEFINITION FIDELITY (this is why the module monkeypatches instead of reimplementing)
-----------------------------------------------------------------------------------
`first_win` is NOT "the level counter went up". Per
`experiment_4605.run_variant_attempt` it is
`solved = reproduction_gate.reproduced and reached_level >= claimed >= 1`, i.e. a
level-up that REPLAYS offline through `arc_solver_kit.reproduce`. Reimplementing that
loop is exactly the "READ, DO NOT MODEL" failure this project keeps making (two
independent reimplementations of a wrong shape agreed 44/44 with each other and were
both wrong about the system). So this harness calls the REAL
`experiment_4605.run_variant_attempt` verbatim and changes exactly ONE thing: which
proposer `_policy_for_mode` installs. Everything downstream of that -- the action loop,
the level accounting, the reproduction gate, the row schema -- is the committed code.

ARMS (all three on the SAME variant list, per-variant matched)
-------------------------------------------------------------
  llm_off   : `_NoOpProposer` -- reproduces the baseline definition bit-for-bit.
  llm_on_16k: shipped proposer with CARNOT_ARC_INDUCE_N_CTX=16384 -- the PRE-FIX
              generator. This is the CONTENTION CONTROL: same tree, same binary, same
              variants, only the context pool reverted. Without it, an llm_on-vs-llm_off
              result could not distinguish "the fault depressed first-win" from "the LLM
              tier never helps here".
  llm_on_fix: shipped proposer at the shipped default (81920) -- the FIXED generator.

CONCURRENCY
-----------
K worker threads over the variant list, mirroring the competition framework
(`swarm.py:91` starts one Thread per game in ONE process with no pool). ONE llama-server
is pre-launched and all workers reuse it via `_ensure_server`'s health check, which is
the eval topology: one local generator, K concurrent clients. K=1 is deliberately NOT
the default -- it is the blind spot that hid the fault for the entire history of this
measurement.

OUTPUT
------
One JSON row file per (arm, variant) under cells/, written as it completes, each carrying
its own `elapsed_s` so a later analyser can publish `measurement_wall_s` from the row
files rather than from a per-cell wall sum (which undercounts).
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

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO))

OUT = REPO / "results" / "first_win_llm_on_20260727"
CELLS = OUT / "cells"

# The three arms. n_ctx None = LLM off entirely (no server involved).
ARMS = {
    "llm_off": None,
    "llm_on_16k": 16384,
    "llm_on_fix": 81920,
    # *_probe are the SAME conditions run on a DIFFERENT cell set (the cells the llm_off
    # control actually won). They get distinct arm labels so their cells can never pool into
    # the pre-specified variant-1 slice's rate -- selecting cells on the control's outcome is
    # biased by construction, and mixing the two scopes into one rate would launder that bias
    # into the headline. See analyse.py's control_winner_probe for how they are read.
    "llm_on_16k_probe": 16384,
    "llm_on_fix_probe": 81920,
    # ADDED 2026-07-27 (adversarial review, FATAL finding: "the treatment was never
    # applied"). Every LLM-on arm above turned out to be BIT-IDENTICAL to its matched
    # llm_off control -- first_win, actions, reached_level and actions_to_first_levelup all
    # equal on 74/74 cells -- because `induction_attempts_planned` was 0 in 174/174 rows:
    # the generator answered (327 calls, 234 responses) and every induced world model was
    # then REJECTED by a POST-generation trust gate, so no plan was ever installed. With the
    # generator's output discarded on every cell, no generator state could move first_win,
    # and p=1.0 / CI [0,0] were arithmetic identities rather than measurements.
    #
    # This arm is the ONLY configuration in which the operator's question is answerable: it
    # is the fixed 81920 generator PLUS CARNOT_ARC_TRUST_METRIC=cell_recall, the project's
    # own named lever for that gate (arc_competition_agent.py:5869-5872 describes exact-match
    # as reading ~0 for an imperfect-but-useful induced model, making induce->plan a no-op).
    # SCOPE LIMIT, measured not assumed: the metric switch only reaches the `else` branch,
    # so it can affect at most the 13 of 25 cells whose skip reason was
    # `world_model_accuracy_below_threshold`. The 11 `hidden_state_trust_below_threshold`
    # cells take the HIDDEN_STATE_GAME_IDS branch, which ignores CARNOT_ARC_TRUST_METRIC
    # entirely, and lp85 failed at `proposer_failed`.
    "llm_on_fix_cellrecall": 81920,
    # ADDED 2026-07-27, second pass. The cell_recall arm above produced a result that
    # CONTRADICTS the lever's own documentation: on lp85 the induced model scored
    # verify_accuracy=0.92 -- comfortably ABOVE the 0.5 trust threshold -- and was gated out
    # anyway because verify_cell_recall was 0.0. So on this corpus cell_recall is STRICTER
    # than the shipped `exact` default, not looser, and it gated out the ONE attempt that
    # would have cleared the default. That makes "planned is 0 under every configuration" an
    # unsupported generalisation from a single arm run with the wrong lever.
    #
    # This arm is the SHIPPED DEFAULT metric (`exact`) at the fixed n_ctx, with the new
    # per-attempt gate diagnostics recording. It is the direct test of whether the treatment
    # is reachable on the path that actually ships. NO env lever: not in ARM_ENV.
    "llm_on_fix_diag": 81920,
}

# Arms that additionally flip an env lever. Kept as a separate table (rather than baked into
# install_arm) so the five ORIGINAL arms are byte-identical in behaviour to the run that
# produced the committed cells -- an arm not listed here sets nothing.
ARM_ENV = {
    "llm_on_fix_cellrecall": {"CARNOT_ARC_TRUST_METRIC": "cell_recall"},
}

_LOCAL = threading.local()


# ---------------------------------------------------------------- server control


def _gguf() -> str:
    import glob

    hits = glob.glob(
        os.path.expanduser(
            "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF/"
            "snapshots/*/Qwen3.5-9B-Q4_K_M.gguf"
        )
    )
    if not hits:
        raise SystemExit("blocked_generator_gguf_missing")
    return sorted(hits)[0]


def launch_server(n_ctx: int, port: int, gpu: int) -> dict:
    """Launch the generator through the SHIPPED launch path (LocalGGUFProposer._ensure_server)
    rather than a hand-built command line.

    This matters: exp5866 priced the VRAM envelope with a hand-built argv, which cannot
    exercise a change to LocalGGUFProposer because it never executes it. Going through
    _ensure_server means the arm under test is the code that actually ships.

    CARNOT_LLAMA_SERVER is set to the CUDA binary and CUDA_VISIBLE_DEVICES to the target
    card, which is resolver priority 1 (`_generator_server_and_env`) -- the same priority
    the Kaggle kernel uses for its bundled CUDA binary. Priority 2's headroom guard is
    deliberately bypassed: that guard would silently fall back to the AMD iGPU HIP build
    if the card looked busy, and a prior lane found it CAN look busy because a healthy
    CUDA server is itself holding the VRAM the guard wants. Device is then VERIFIED from
    per-PID residency, never from the env var having been set.
    """
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    cuda_bin = os.path.expanduser("~/.cache/llama.cpp-master/build/bin/llama-server")
    os.environ["CARNOT_LLAMA_SERVER"] = cuda_bin
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["CARNOT_ARC_INDUCE_N_CTX"] = str(n_ctx)
    os.environ["CARNOT_ARC_PROPOSER_PORT"] = str(port)

    p = LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP",
        model_path=_gguf(),
        mtp=True,
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=4096,
        timeout=600,
        port=port,
        n_gpu_layers=999,
    )
    assert p.n_ctx == n_ctx, f"n_ctx override did not take: {p.n_ctx} != {n_ctx}"
    t0 = time.time()
    ok = p._ensure_server()
    info = {
        "requested_n_ctx": n_ctx,
        "port": port,
        "server_binary": cuda_bin,
        "server_binary_is_cuda_build": "/build/bin/" in cuda_bin,
        "launch_ok": bool(ok),
        "launch_s": round(time.time() - t0, 2),
        "pid": getattr(p._proc, "pid", None),
    }
    if not ok:
        info["blocked"] = "blocked_generator_server_launch_failed"
        return info
    info["props"] = read_props(port)
    info["device"] = verify_device(info["pid"], gpu)
    return info


def read_props(port: int) -> dict:
    """READ the server's own /props. Do not compute what the server will tell you."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=10) as r:
            d = json.loads(r.read().decode())
        slots = d.get("total_slots")
        dp = d.get("default_generation_settings") or {}
        return {
            "total_slots": slots,
            "n_ctx_reported": dp.get("n_ctx"),
            "n_ctx_per_seq": (d.get("n_ctx_per_seq") if "n_ctx_per_seq" in d else None),
            "model_path": str(d.get("model_path") or "")[-60:],
        }
    except Exception as exc:
        return {"props_error": repr(exc)[:200]}


def verify_device(pid: int | None, want_gpu: int) -> dict:
    """Device verdict from PER-PID VRAM RESIDENCY, not from the env var being set.

    A prior lane established that setting CUDA_VISIBLE_DEVICES is NOT evidence: the
    resolver can silently launch the AMD iGPU HIP build. The only proof is that this
    process id shows up holding memory against the intended card's UUID.
    """
    out = {"pid": pid, "intended_gpu": want_gpu}
    try:
        uuid_map = {}
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        for line in r.stdout.strip().splitlines():
            idx, uu = (x.strip() for x in line.split(","))
            uuid_map[uu] = int(idx)
        r2 = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,gpu_uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
        rows = []
        for line in r2.stdout.strip().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue
            rows.append(
                {
                    "pid": int(parts[0]),
                    "used_mib": int(parts[1].split()[0]),
                    "gpu_index": uuid_map.get(parts[2], -1),
                }
            )
        out["compute_apps"] = rows
        mine = [r for r in rows if r["pid"] == pid]
        out["my_rows"] = mine
        if mine and all(r["gpu_index"] == want_gpu for r in mine):
            out["verdict"] = f"CONFIRMED_GPU{want_gpu}_BY_PER_PID_RESIDENCY"
            out["my_vram_mib"] = sum(r["used_mib"] for r in mine)
        elif mine:
            out["verdict"] = "WRONG_DEVICE"
        else:
            out["verdict"] = "NOT_RESIDENT_ON_ANY_NVIDIA_GPU_possible_igpu_fallback"
        r3 = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        out["gpu_totals"] = r3.stdout.strip().splitlines()
    except Exception as exc:
        out["verdict"] = "UNVERIFIED"
        out["error"] = repr(exc)[:200]
    return out


def _pids_on_port(port: int) -> list[int]:
    """Every pid LISTENING on `port`, resolved from `ss` rather than from a name pattern.

    WHY NOT `pkill -f`: the pattern would match this harness's own command line. Port
    ownership is the precise, safe identifier -- it names exactly the process serving this
    arm and nothing else.
    """
    out: list[int] = []
    try:
        r = subprocess.run(["ss", "-ltnpH"], capture_output=True, text=True, timeout=20)
        for line in r.stdout.splitlines():
            if f"127.0.0.1:{port}" not in line:
                continue
            for tok in line.split("pid=")[1:]:
                digits = ""
                for ch in tok:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    out.append(int(digits))
    except Exception:
        return out
    return sorted(set(out))


def _terminate(pid: int) -> str:
    try:
        os.kill(pid, 15)
    except OSError as exc:
        return f"absent({exc.errno})"
    for _ in range(30):
        time.sleep(1)
        try:
            os.kill(pid, 0)
        except OSError:
            return "SIGTERM"
    try:
        os.kill(pid, 9)
        time.sleep(2)
        return "SIGKILL"
    except OSError:
        return "SIGTERM_late"


def kill_server(pid: int | None, port: int | None = None) -> dict:
    """Teardown by EXPLICIT pid AND by PORT OWNERSHIP. Never `pkill -f <pattern>`.

    WHY THE PORT SWEEP EXISTS (measured leak, 2026-07-27). The recorded pid is the one
    LocalGGUFProposer's first `_ensure_server()` spawned. But every worker thread calls
    `_ensure_server()`, and if the server dies mid-run a worker RELAUNCHES it -- producing a
    long-lived server this harness never recorded. That happened on the llm_on_16k arm: the
    recorded pid 2271222 was already gone (teardown logged
    ProcessLookupError), while pid 2279120 was still holding 11818 MiB on GPU 1 and still
    listening on that arm's port. The next arm then could not fit its own server in the
    remaining VRAM and honestly reported blocked_generator_server_launch_failed after waiting
    600 s. Killing by recorded-pid alone is therefore not sufficient; whatever currently owns
    the arm's port must go too.
    """
    result: dict = {"recorded_pid": pid, "port": port}
    if pid:
        result["recorded_pid_outcome"] = _terminate(int(pid))
    strays = [p for p in (_pids_on_port(int(port)) if port else []) if p != pid]
    result["port_owner_strays_found"] = strays
    result["port_owner_outcomes"] = {str(p): _terminate(p) for p in strays}
    result["killed"] = bool(
        result.get("recorded_pid_outcome") in {"SIGTERM", "SIGKILL", "SIGTERM_late"}
    ) or bool(strays)
    result["port_clear_after"] = _pids_on_port(int(port)) == [] if port else None
    return result


# ---------------------------------------------------------------- the measurement


def install_arm(exp4605, arm: str) -> None:
    """Replace exp4605._policy_for_mode with a proposer-parameterised twin.

    The body is otherwise line-for-line the original (target_levels / value_weight read
    from the SAME SUBMITTED_* helpers), so the measured agent stays the parity-gated
    submitted config. `proposer=None` is the meaningful change: it makes E3AgentPolicy
    lazily build its own SHIPPED LocalGGUFProposer via `_proposer()`, which is the object
    the scored kernel uses -- rather than the NoOp stub the baseline installed.
    """
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    def _policy_for_mode(mode: str, game: str):
        if mode == "bare":
            # The bare control is LLM-off in the committed code and stays LLM-off here;
            # it is not one of the three arms under test.
            return E3AgentPolicy(
                game,
                proposer=exp4605._NoOpProposer(),
                target_levels=1,
                value_head=None,
                value_weight=0.0,
                candidate_router=None,
                navigation_cost_tiebreak=False,
            )
        proposer = exp4605._NoOpProposer() if arm.startswith("llm_off") else None
        pol = E3AgentPolicy(
            game,
            proposer=proposer,
            target_levels=exp4605._submitted_target_levels(),
            value_weight=exp4605._submitted_value_weight(),
        )
        _LOCAL.policy = pol  # so the caller can read the liveness witness afterwards
        return pol

    exp4605._policy_for_mode = _policy_for_mode


def run_cell(exp4605, arm: str, spec: dict, budget: int) -> dict:
    cell = CELLS / f"{arm}__{spec['variant_signature'].replace('~', '_')}.json"
    if cell.exists():
        return json.loads(cell.read_text())
    _LOCAL.policy = None
    t0 = time.time()
    row: dict = {}
    err = ""
    try:
        row = dict(exp4605.run_variant_attempt("integrated", str(spec["game"]), spec, budget))
    except Exception as exc:  # a crashed cell must be VISIBLE, never an implicit null
        err = repr(exc)[:400]
    elapsed = time.time() - t0
    witness = None
    pol = getattr(_LOCAL, "policy", None)
    if pol is not None:
        try:
            witness = pol.generator_liveness_witness()
        except Exception as exc:
            witness = {"witness_error": repr(exc)[:200]}
    out = {
        "arm": arm,
        "game": str(spec["game"]),
        "variant": int(spec["variant"]),
        "variant_signature": spec["variant_signature"],
        "elapsed_s": round(elapsed, 3),
        "cell_error": err,
        "row": row,
        "first_win": bool(row.get("first_win")) if row else None,
        "reached_level": row.get("reached_level") if row else None,
        "actions": row.get("actions") if row else None,
        "actions_to_first_levelup": row.get("actions_to_first_levelup") if row else None,
        "liveness_witness": witness,
    }
    cell.parent.mkdir(parents=True, exist_ok=True)
    cell.write_text(json.dumps(out, indent=1, default=str))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--k", type=int, default=4, help="concurrent workers (eval shape)")
    ap.add_argument("--variants", default="1,2,3,4")
    ap.add_argument("--games", default="", help="comma list; default = all public games")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--port", type=int, default=8951)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    CELLS.mkdir(parents=True, exist_ok=True)
    # Match the baseline's env exactly: the 4605 artifact's integrated rows carry
    # depth_reached / actions_to_second_levelup, which only exist when GATE_DEEPEN is on.
    os.environ["CARNOT_ARC_GATE_DEEPEN"] = "1"
    os.environ["CARNOT_ARC_GATE_VARIANT_IDS"] = args.variants
    os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
    # Per-arm env levers (see ARM_ENV). Explicitly CLEARED for every other arm so a stale
    # ambient value cannot silently change what an arm measures -- the declared-vs-actual
    # failure class this whole measurement exists to close.
    for _var in {k for env in ARM_ENV.values() for k in env}:
        os.environ.pop(_var, None)
    for _var, _val in ARM_ENV.get(args.arm, {}).items():
        os.environ[_var] = _val
        print(f"[arm-env] {_var}={_val}", flush=True)

    import carnot.experiment_4605_live_integration_scored_agent as exp4605

    games = (
        [g.strip() for g in args.games.split(",") if g.strip()]
        if args.games
        else exp4605._public_games(REPO)
    )
    variant_ids = [int(v) for v in args.variants.replace(",", " ").split()]
    specs = exp4605.variant_specs(games, variant_ids)

    meta = {
        "arm": args.arm,
        "k_concurrency": args.k,
        "n_ctx": ARMS[args.arm],
        "games": games,
        "variant_ids": variant_ids,
        "budget": args.budget,
        "n_cells": len(specs),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "submitted_target_levels": exp4605._submitted_target_levels(),
        "submitted_value_weight": exp4605._submitted_value_weight(),
        "deepen_enabled": exp4605._deepen_enabled(),
    }

    server = None
    if ARMS[args.arm] is not None:
        server = launch_server(int(ARMS[args.arm]), args.port, args.gpu)
        meta["server"] = server
        if not server.get("launch_ok"):
            (OUT / f"run_{args.arm}{args.tag}.json").write_text(
                json.dumps({**meta, "blocked": server.get("blocked")}, indent=1, default=str)
            )
            print("BLOCKED:", server)
            return 2
        print(
            f"[server] n_ctx={ARMS[args.arm]} pid={server['pid']} "
            f"props={server['props']} device={server['device'].get('verdict')} "
            f"vram={server['device'].get('my_vram_mib')}"
        )
    else:
        meta["server"] = {"llm_off": True, "note": "no generator involved in this arm"}

    install_arm(exp4605, args.arm)

    t0 = time.time()
    done = 0
    rows = []
    with ThreadPoolExecutor(max_workers=args.k) as ex:
        futs = {ex.submit(run_cell, exp4605, args.arm, s, args.budget): s for s in specs}
        for f in as_completed(futs):
            rows.append(f.result())
            done += 1
            if done % 4 == 0 or done == len(specs):
                print(
                    f"[{args.arm}] {done}/{len(specs)} wall={time.time() - t0:.0f}s "
                    f"wins={sum(1 for r in rows if r.get('first_win'))}",
                    flush=True,
                )

    meta["wall_s"] = round(time.time() - t0, 2)
    meta["measurement_wall_s_from_rows"] = round(sum(float(r["elapsed_s"]) for r in rows), 2)
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if server:
        meta["server_healthy_after"] = _health(args.port)
        meta["props_after"] = read_props(args.port)
        meta["teardown"] = kill_server(server.get("pid"), args.port)
        meta["gpu_after_teardown"] = (
            subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()
        )
    (OUT / f"run_{args.arm}{args.tag}.json").write_text(json.dumps(meta, indent=1, default=str))
    print(
        f"DONE {args.arm}: cells={len(rows)} wins={sum(1 for r in rows if r.get('first_win'))} "
        f"wall={meta['wall_s']}s"
    )
    return 0


def _health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return b"ok" in r.read()
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
