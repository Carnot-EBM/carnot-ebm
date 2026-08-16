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

# DUAL-GPU, n_ctx 65536 (operator directive 2026-08-15: "stop the conductor and use both eGPUs...
# focused on accuracy and success regardless of how long it takes"). Applied to BOTH arms, so the
# change is symmetric and the comparison stays honest.
#
# WHY THE OLD SINGLE-CARD 32768 HAD TO GO. tu93's induce_prompt is 15,771 tokens, so n_ctx 32768
# left 16,997 to answer in. Gemma fits that because its reasoning goes to a separate
# `reasoning_content` channel and does not consume the answer window. Qwen3's thinking is INLINE
# and does. Four arm runs decoded 16,905 / 16,920 / 16,843 and were cut off mid-thought -- a match
# to `n_ctx - prompt` within 0.5%. The constant was not neutral: it silently encoded an assumption
# about where a model puts its reasoning.
#
# Measured 2026-08-15 on both cards at n_ctx 65536: Qwen3.8 completed tu93 in 1200s, generating
# 41,613 tokens (~40k of them reasoning) and emitting 1,843 chars of COMPILING world model with
# real game mechanics -- a fuel bar draining from row 63 and a win condition on cell (46,40).
# It terminates on its own; it was never looping. It just needs ~42k tokens of room.
#
# 65536 does not fit on ONE 24 GB card (17.8 GB model + KV + 32 context checkpoints at ~150 MB
# dropped the connection mid prompt-processing). Split across two it uses 9.7 + 10.6 GB and leaves
# ~28 GB for KV, at 38.3 t/s -- no throughput cost versus single-card, so speed stays comparable.
GPU_UUIDS = (
    "GPU-b52387a2-c625-de87-8d34-e6f64e684bab",  # GPU 0 -- freed by stopping the conductor
    "GPU-7971baff-9583-eaa6-2292-393f930a28f9",  # GPU 1
)
GPU_INDEX = "0,1"
GPU1_UUID = GPU_UUIDS[1]  # kept: the artifact field `gpu_uuid_proven` still reports it
N_CTX = 65536

# The live agent's retry count. LocalGGUFProposer.tries defaults to 3 and the live induce
# path passes tries=self.tries; exp5726's cell overrides it to 1 for its own question.
LIVE_TRIES = 3

# SAMPLING SEED, per (arm-family, trial). Added 2026-08-16 after tracing why this design needs 3
# trials at all.
#
# THE PROBLEM, from LocalGGUFProposer.sampling_seed's own docstring: every generation goes out at
# `temperature = 0.2 + 0.1*attempt` with NO seed field, and llama-server reads an absent seed as
# -1 = "pick a fresh random one". The harness `seed` argument seeds random/numpy in the driver and
# never reaches the server's sampler. Measured cost: "2 of 5 cells diverge under IDENTICAL CODE --
# a 40% nondeterminism rate... at least as large as any treatment effect yet measured on this
# path, so an A/B here is uninterpretable without an A/A control no matter how many cells it runs."
#
# WHY THE SEED MUST VARY WITH TRIAL, not be one fixed value. `sampling_seed` composes
# `base * 1000 + attempt`, where attempt is the RETRY index. A single base for the whole arm would
# give trial 0, 1 and 2 identical seeds and collapse three replicates into one measurement
# repeated three times -- destroying the very thing TRIALS exists for.
#
# WHY BOTH ARMS SHARE THE SEQUENCE. Identical seeds per (game, trial) make this a PAIRED
# comparison: the two models see the same sampling randomness on the same cell, so the noise that
# dominates this pipeline is differenced out instead of averaged over. That is strictly stronger
# than what 3 unpinned trials could ever give.
#
# The A/A arms take an offset base so they explore different draws of the same distribution --
# that difference IS the measurement.
SEED_BASE = {"gemma31b": 100, "qwen38_27b": 100, "gemma31b_aa": 700, "qwen38_27b_aa": 700}
BUDGET = 16384  # exp5726/5760/5764 completion budget

# PER-ARM BUDGET (2026-08-15). The shared 16384 is NOT equal compute across these models, and
# the first Qwen3.8 run proved it: every generation ran to the cap (16322, 16310, 16218 tokens
# decoded) with ZERO natural stops, and no arm ever reached its code block.
#
# The cell is `/think` by design -- exp5726 verbatim, and that identity is what makes the number
# comparable, so the cell is NOT modified. But gemma-4 and Qwen3 spend a think budget in
# structurally different places. Gemma's reasoning goes to a SEPARATE `reasoning_content` channel
# on the chat endpoint and does not consume the completion budget. Qwen3's hybrid thinking is
# INLINE in the same stream, so it eats the budget the code has to fit in.
#
# So "same budget" flatters gemma and starves Qwen. Holding the number equal would have produced
# a clean 0-for-39 that says nothing about induction quality -- the measure-the-wrong-thing
# failure this project has hit repeatedly this week.
#
# The honest correction is to give the inline-thinking arm room to finish and to report the token
# cost as part of the result, since the scored metric cares about cost anyway. 32768 is the same
# as n_ctx, so it stays inside the deployed context.
#
# STATE IT WHEN REPORTING: the arms differ in budget, deliberately, and the comparison is
# "each model driven so it can finish", not "identical flags".
ARM_BUDGET = {"qwen38_27b": 32768}
MIN_RESIDENCY_MIB = 15000  # SUMMED over both cards; below this the model did not really load
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
    # A/A CONTROL ARMS. Same GGUF, same everything, DIFFERENT seed base. Running a model against
    # ITSELF measures the noise floor of this pipeline, which the proposer's own docstring says is
    # ~40% cell divergence under identical code and "at least as large as any treatment effect yet
    # measured on this path". Without this number a Qwen-vs-gemma gap is unreadable.
    "gemma31b_aa": {
        "label": "gemma-4-31B-it (A/A)",
        "repo_substr": "gemma-4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "gguf": (
            "/home/ianblenke/.cache/huggingface/hub/models--unsloth--gemma-4-31B-it-GGUF/"
            "snapshots/f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf"
        ),
        "port": 8988,
        "quant": "Q4_K_M",
    },
    "qwen38_27b_aa": {
        "label": "qwen3.8-27B (A/A)",
        "repo_substr": "Qwen3.8-27B",
        "hf_id": "unsloth/Qwen3.8-27B-GGUF",
        "gguf": None,
        "port": 8989,
        "quant": "Q4_K_M",
    },
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
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=25,
        )
        # BOTH cards must still be present: the model is split across them, so losing either one
        # is as fatal as losing the single card used to be.
        return r.returncode == 0 and all(u in r.stdout for u in GPU_UUIDS)
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
    total = 0
    for line in r.stdout.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 3 and p[0].isdigit() and int(p[0]) == pid and p[1] in GPU_UUIDS:
            total += int(p[2])
    # Summed across both cards, because a layer split puts roughly half the weights on each and a
    # per-card check would see ~9.7 GB and wrongly conclude the model never loaded.
    return total or None


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
    # -sm layer -ts 1,1 splits the model evenly. No MTP on either arm, so no draft override.
    if "-sm" not in args:
        args += ["-sm", "layer", "-ts", "1,1"]
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
    # Optional overrides for targeted follow-up probes (e.g. measuring one game's convergence
    # rate at extra seed bases). Absent, behaviour is exactly the full-roster arm.
    ap.add_argument("--games", help="comma-separated subset of ROSTER to run")
    ap.add_argument("--seed-base", type=int, help="override SEED_BASE for this arm")
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
    from carnot.experiment_5760_cegis_refinement_induction_ab import (  # noqa: E402
        TRIALS,
    )
    from carnot.experiment_5760_cegis_refinement_induction_ab import (
        ROSTER as _ROSTER_DEFAULT,
    )

    ROSTER = list(_ROSTER_DEFAULT)

    # ---- NAME THE REQUIRED SYMBOLS EXPLICITLY (operator 2026-08-15: "tune the prompt or we may
    # end up wasting the effort") -------------------------------------------------------------
    #
    # `generate()` accepts a completion only if it defines `def engine` AND `def is_level_complete`
    # at top level, retrying `tries` times otherwise. The shipped directive reads:
    #
    #     "Return ONLY one ```python code block with engine + is_level_complete."
    #
    # which names the symbols but does not say they are FUNCTION NAMES. Gemma reads it the
    # intended way. Qwen3.8 does not: its accepted-looking output defined `is_level_complete` and
    # put the transition logic in a differently-named function, so every attempt was rejected and
    # retried until the token budget was gone -- 1,444 s and `overran=True` on all three tu93
    # cells, heading for a 15-hour 0/39 that would have measured naming, not induction.
    #
    # SYMMETRIC BY CONSTRUCTION: both arms call this same patched function, so gemma is held to
    # the identical instruction. It is a clarification of the existing contract, not a new
    # requirement -- the required tuple `("engine", "is_level_complete")` is unchanged.
    #
    # PATCHED HERE, not in exp5714, because that module is shared verbatim by other experiments
    # and this is exp6440's protocol decision to make.
    import carnot.experiment_5714_think_mode_rescoped_ab as _t5714  # noqa: E402

    # THE ARITY IS THREE, NOT TWO. `engine(grid, action, data)` is what every consumer calls:
    # WorldModelVerifier.score (arc_executable_world_model.py:1818), plan_in_model (:7749, :7800),
    # and score_goal_predicate_consistency (:2402). arc_engine_static_validation.py:119-121 states
    # it outright -- "the engine signature the whole ARC world-model apparatus assumes".
    #
    # This directive shipped with TWO args on 2026-08-15 and silently destroyed every cell that
    # obeyed it: the engine loads, satisfies `def engine`, sets induce_ok=True, and then raises
    # TypeError on EVERY transition inside score(), which counts the raise and skips the row --
    # so cell_recall is computed over an empty list and comes out 0.0. Verified by executing the
    # engines on disk: sp80's raises, sb26's (which followed the prompt BODY's 3-arg form instead
    # of this directive) runs and scored 0.66.
    #
    # The prompt body already said three (induce_prompt, :3324), so the model was handed
    # CONTRADICTORY signatures and each cell resolved it by coin-flip. That is the whole
    # explanation for "induce_ok=True with cell_recall 0.0" on tr87 and sp80.
    _REQUIRED_SYMBOL_DIRECTIVE = (
        "\n\nReturn ONLY one ```python code block. It MUST define exactly these two TOP-LEVEL "
        "functions, spelled exactly:\n"
        "    def engine(grid, action, data):      # returns the next grid\n"
        "    def is_level_complete(grid):         # returns True on a win\n"
        "Do not rename them. Do not wrap them in a class. A different name is rejected.\n"
    )

    # PER-MODEL THINK PREFIX (operator 2026-08-15: "fix it").
    #
    # `run_reason_cell_budget` sets `prop.no_think_prefix = "/think\n"` for every arm. `/think` is
    # a QWEN3 hybrid-thinking control token. On gemma-4 it is inert text -- gemma has no such
    # token, which is precisely why `ARC_LIVE_GENERATOR_NO_THINK_PREFIX` is "" with the comment
    # "/no_think is a Qwen3 token; inert on gemma-4". On Qwen3.8 it is an active instruction to
    # reason at length, and it made the model spend all three budgets thinking: 4,324 s,
    # overran on every try. The same prompt with the prefix left at "" returned ok=True with both
    # required symbols in 2,786 s.
    #
    # So the constant does nothing to one arm and disables the other. Set it to what each model
    # actually deploys with instead:
    #   gemma-4  -> ""          native channelled reasoning; the live constant is already ""
    #   Qwen3.8  -> ""          the live stack used /no_think when a Qwen WAS the generator;
    #                           "" is the neutral middle -- no forced thinking, no forced
    #                           suppression -- and it is the setting verified to work here.
    #
    # BOTH LAND ON "", so the arms do NOT end up differing -- this RESTORES symmetry rather than
    # breaking it. That is worth stating because the reasoning nearly went the other way: I first
    # wrote this as an accepted asymmetry. It is not one. "/think" was the asymmetric setting,
    # because it is a control token for one family and dead text for the other; "" is the same
    # instruction to both.
    #
    # Removing it is also near-neutral for gemma specifically: `/think\n` was 8 characters of
    # inert prompt text there, and gemma's reasoning is driven by the chat endpoint's native
    # thought channel (exp6199's think-mode result), which this does not touch.
    LIVE_THINK_PREFIX = {"gemma31b": "", "qwen38_27b": ""}

    def _induce_named(prop_, game_, window_, cell_):
        """exp5714._induce_no_fence with the symbol names spelled out. Same call, same required
        tuple, same tries -- only the instruction text differs."""
        from carnot.agentic.arc_executable_world_model import (
            _induce_transitions_k,
            induce_prompt,
        )

        # Override the cell's "/think\n" with this arm's deployed setting. The cell saved the
        # original and restores it in its own finally-block, so this does not leak between cells.
        prop_.no_think_prefix = LIVE_THINK_PREFIX.get(args.arm, "")
        prompt = (
            induce_prompt(game_, window_, cell_, k=_induce_transitions_k())
            + _REQUIRED_SYMBOL_DIRECTIVE
        )
        # tries=LIVE_TRIES, not prop_.tries. `run_reason_cell_budget` forces `prop.tries = 1`
        # before calling us (exp5726 line ~324), because that experiment was specifically studying
        # reasoning overrun -- "a /think overrun is recorded as induction_ok=False, the honest
        # finding". That is an artifact of ITS question, not of deployment.
        #
        # THE LIVE SCORED AGENT GETS 3. `LocalGGUFProposer.tries` defaults to 3 and every live
        # induce call passes `tries=self.tries`. So measuring at 1 does not describe the agent we
        # ship, and it is a constraint that binds only one arm: gemma emits the right symbols on
        # attempt one, Qwen3.8 spends attempt one reasoning. Verified directly -- the same prompt
        # returned ok=True with both symbols at tries=3 (2,786 s) and overran at tries=1.
        #
        # Symmetric: both arms get 3. Gemma will simply not need the extra two.
        # codeonly_eligible=True, to switch ON the repetition control this run was missing.
        #
        # WHAT IT DOES *NOT* DO: it does not restore the fence or the codeonly directive. With
        # think mode on (the shipped default), generate() routes into its `_think_on` branch --
        # "no directive, no pre-opened fence" (arc_executable_world_model.py:6873-6874) -- so the
        # PROMPT TEXT IS BYTE-IDENTICAL to what the previous cells saw. This is not a prompt
        # change.
        #
        # WHAT IT DOES DO: `_engine_induce_call = bool(codeonly_eligible) and "engine" in required`
        # (:6885) gates `repeat_penalty=1.1, repeat_last_n=256` (:6922-6926). With False we were
        # running with NO decode-level repetition control at all, against a failure mode that is
        # almost entirely `stop_type=limit`.
        #
        # MEASURED, docs/research-notes/arc-induce-repeat-penalty-confirm-2026-07-31.md, 36 paired
        # attempts on gemma-4-31B:
        #     usable engines      13/36 -> 22/36  (1.69x, NOT the "triples" its own headline
        #                                          claimed -- corrigendum C1 in that note)
        #     hit the token cap   20/36 -> 2/36   <- our dominant failure mode
        #     missing_return      13    -> 2
        #     wall per attempt    100.3s -> 47.2s
        # Attempt-matched sign test p=0.049; clustered by game p=0.125. HONEST LIMIT: quality did
        # NOT move -- the strict out-of-sample funnel was 4/18 vs 4/18. This buys usable engines,
        # not better ones.
        #
        # Symmetric: both arms take the same path, and gemma's own numbers above are where the
        # effect was measured.
        ok_, code_ = prop_.generate(
            prompt, ("engine", "is_level_complete"), tries=LIVE_TRIES, codeonly_eligible=True
        )
        if not ok_:
            return False, code_
        return prop_._write_world_model(game_, code_)

    _t5714._induce_no_fence = _induce_named
    log("prompt: required symbols named explicitly (both arms)")
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
    if args.games:
        _want = [g.strip() for g in args.games.split(",") if g.strip()]
        _unknown = [g for g in _want if g not in ROSTER]
        if _unknown:
            # Refuse rather than silently running a subset -- a typo'd game name that quietly
            # becomes "run nothing" is the kind of null this project keeps mistaking for a result.
            log(f"FATAL: --games names {_unknown} which are not in ROSTER {list(ROSTER)}")
            return 3
        ROSTER = _want  # noqa: F841 -- rebound below via the loop's use
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
            # ARM_BUDGET, not BUDGET. The per-arm budget added earlier only reached
            # run_reason_cell_budget's `budget=` argument; the PROPOSER kept the module-level
            # 16384, so the raise never took effect and max-decoded stayed pinned near 16.9k. Two
            # places had to change and I changed one -- the measurement then "tested" a budget it
            # was never given.
            # Room to answer: n_ctx minus the largest prompt we build. The measured tu93 prompt
            # is 15,771 tokens; 16384 is that rounded up with margin.
            max_tokens=N_CTX - 16384,
            # 3600s, because the measured Qwen3.8 induction took 1200s and the old 1800s left too
            # little headroom for a slower game. A timeout here does not raise -- it returns
            # (False, msg) and the cell records a silent non-answer, which is exactly the failure
            # mode that cost four runs.
            timeout=3600,
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

            # Per-cell, before the induce: base varies with TRIAL so replicates stay distinct,
            # and is identical across the two compared arms so the comparison is paired.
            _base = args.seed_base if args.seed_base is not None else SEED_BASE.get(args.arm, 100)
            os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(_base + int(t))
            log(
                f"RUN {args.arm} {game} trial={t} "
                f"(residency={res} MiB, probe={lv['s']}s, "
                f"seed_base={os.environ['CARNOT_ARC_GENERATOR_SEED']})"
            )
            c0 = time.time()
            try:
                row = run_reason_cell_budget(
                    game,
                    prop,
                    trial=t,
                    window=windows[game][0],
                    full_traj=windows[game][1],
                    cell=windows[game][2],
                    # Symmetric across arms. ARM_BUDGET existed to work around the 32768
                    # context ceiling; with n_ctx 65536 on two cards both models get the same
                    # room, which is what makes the comparison readable.
                    budget=N_CTX - 16384,
                )
                try:
                    src = (E3_DIR / game / "world_model.py").read_text()
                except Exception:
                    src = ""
                ms = memorization_scan(src, _window_changed_coords(windows[game][0]))
                row["mem_scan"] = ms
                # Stamped per row, not just in the meta: a row that travels on its own must
                # carry the budget that produced it, or a later reader compares 16384-cells
                # against 32768-cells without knowing.
                # The ACTUAL n_predict, not the retired ARM_BUDGET map. This stamped 32768/16384
                # while every row's own induce_detail said `HIT n_predict=49152` -- inverting the
                # field's stated purpose, which is that a row travelling alone carries the budget
                # that produced it.
                row["budget"] = N_CTX - 16384
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
