#!/usr/bin/env python3
"""ONE CELL of the treatment-activation pre-flight for the PHASE-2 WIRED induce treatment.

THE QUESTION. Phase 2 wired two changes into the live code-only induce path:
``repeat_penalty=1.1`` (+ ``repeat_last_n=256``) and a defect gate that issues ONE PLAIN
re-ask. Phase 1 showed both change the INDUCE OUTCOME (valid engines 13/36 -> 22/36). This
probe asks the strictly downstream question the banked-levels grid actually depends on: does
that reach the LIVE AGENT'S ACTIONS at all? If the two arms emit the same action stream, every
endpoint computed from those actions is identical by construction and the grid is dead on
arrival regardless of how much the induce payload changed.

WHY THIS PROBE IS CLEANER THAN THE 2026-07-30 COMPOSITE ONE, and it is worth being explicit
because that probe's hard-won defences are mostly UNNEEDED here rather than merely omitted:

  * that probe compared two CHECKOUTS (8441055c0 vs HEAD) served from two directories, so it
    needed a backported sampler seed, an externally-defined trace recorder, a written-in arm
    config literal, and a per-cell asset stamp -- and it still shipped a confound (two
    gitignored assets absent from the worktree changed the frontier expansion order, and
    NEITHER A/A floor could see it because each floor held the asset axis fixed).
  * here BOTH arms are the SAME COMMIT, the SAME repo path, the SAME assets, the SAME
    interpreter and the SAME server process. The only thing that differs is two environment
    variables read inside `induce()`. There is no arm axis for an asset, a shim or a commit
    to hide in, so the class of confound that corrupted that grid cannot arise.

THE ARMS:
  ctl   CARNOT_ARC_INDUCE_REPEAT_PENALTY=1.0 + CARNOT_ARC_INDUCE_DEFECT_REASKS=0.
        Verified against the source, not assumed: `_induce_repeat_penalty()` returning 1.0
        makes the wiring skip BOTH payload keys entirely (`if _rp != 1.0`), so the request
        body is byte-identical to the pre-wiring one, and 0 re-asks restores accept-first.
  trt   the shipped defaults (1.1 / 256 / one re-ask). THE WIRED CODE, unmodified.
  ctlb  `ctl` again -- the A/A NOISE FLOOR, and it is not garnish. Measured on this box
        2026-07-30, a provably-same-code A/A at f9a458e87 still diverged on 1 of 2 cells, so
        seeding the sampler is NECESSARY but NOT SUFFICIENT. Perturbation that is not
        differenced against a floor is not attributable to anything.

WHY max_inductions=1. The treatment fires inside `induce()`. One induction means it fires
exactly once per cell, so a divergence has one place it can have come from. It also fixes the
attribution boundary: `explore_budget=24` transitions are collected BEFORE the first induce, so
any divergence at an action index earlier than the first induction is, by construction, NOT this
treatment -- it is harness noise or upstream drift. That distinction is computed in the scorer
(`first_induction_action_index`), because the 2026-07-30 probe found 3 of its 4 "attributable"
cells diverged pre-induction and the raw perturbation rate therefore overstated the effect.

SETUP FACTS THAT PRODUCE A FAKE RESULT IF IGNORED (all verified on this box):
  * gemma-4-31B at the shipped n_ctx=81920 does not fit a 24 GiB card; the loader silently
    falls through to the iGPU HIP build, where the agent runs LLM-OFF while REPORTING LLM-ON.
    Hence n_ctx=32768 + ffn_cpu_layers=0, and a HARD BLOCK if the bound binary is the HIP one.
  * this parent must hold no CUDA context of its own (~396 MiB is enough to tip the VRAM
    guard), hence CUDA_VISIBLE_DEVICES="" + JAX_PLATFORMS=cpu.
  * card identity is READ from /proc/<pid>/exe and per-PID VRAM, never inferred from
    CUDA_VISIBLE_DEVICES, which records only what we asked for.
  * a non-default port: 8919 is the default and a stale server there is silently adopted.
  * CARNOT_ARC_E3_DIR is redirected to a per-cell scratch dir and asserted EMPTY. All 25
    public games have banked engines in the canonical store; loading one would make the arms
    agree for reasons that have nothing to do with the treatment. The canonical store is
    EVIDENCE and is never written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time

MAIN_REPO = "/home/ianblenke/github.com/ianblenke/carnot"
HERE = os.path.dirname(os.path.abspath(__file__))

ARM = sys.argv[1]  # ctl | trt | ctlb
GAME = sys.argv[2]
SEED = int(sys.argv[3])

# The ONLY axis that differs between arms. `None` means "leave unset", i.e. take the shipped
# default -- which is what makes `trt` the actual wired code rather than a re-specification of
# it. Writing "1.1"/"1" explicitly for trt would silently pass even if the wiring were reverted.
ARM_ENV = {
    "ctl": {"CARNOT_ARC_INDUCE_REPEAT_PENALTY": "1.0", "CARNOT_ARC_INDUCE_DEFECT_REASKS": "0"},
    "ctlb": {"CARNOT_ARC_INDUCE_REPEAT_PENALTY": "1.0", "CARNOT_ARC_INDUCE_DEFECT_REASKS": "0"},
    "trt": {},
    # trtb -- the TREATMENT-arm A/A replicate. Same empty dict as `trt` on purpose: an A/A
    # replicate that re-specified the treatment explicitly would still pass if the wiring were
    # reverted, which is exactly what the replicate is supposed to be able to notice.
    "trtb": {},
}
if ARM not in ARM_ENV:
    raise SystemExit(f"unknown arm {ARM!r}")

BUDGET = int(os.environ.get("PROBE_BUDGET", "60"))
MAX_IND = int(os.environ.get("PROBE_MAX_IND", "1"))
WALL_S = float(os.environ.get("PROBE_WALL_S", "1200"))
EXPLORE_BUDGET = int(os.environ.get("PROBE_EXPLORE_BUDGET", "24"))

GPU = os.environ["CELL_GPU"]
PORT = int(os.environ["CELL_PORT"])
REPO_SUBSTR = "gemma-4-31B-it"

# A dedicated subtree. `HERE` is shared with a PRIOR session's LLM-on/off A/B, whose cells
# (arms `on`/`off`/`onb`/`offb`) already sit in `HERE/cells`. Nothing here would read them --
# the arm names do not collide -- but two experiments' records in one directory is how a
# census ends up counting somebody else's run, so they are kept apart by construction.
OUTDIR = os.path.join(HERE, "pf", "cells")
OUT = os.path.join(OUTDIR, f"{ARM}__{GAME}__s{SEED}.json")
E3_DIR = os.path.join(HERE, "pf", "e3", f"{ARM}__{GAME}__s{SEED}")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(E3_DIR, exist_ok=True)

# MUST precede the import: E3_DIR is resolved at module-import time.
os.environ["CARNOT_ARC_E3_DIR"] = E3_DIR
os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = GPU
os.environ["CARNOT_ARC_INDUCE_N_CTX"] = "32768"
os.environ["CARNOT_ARC_FFN_CPU_LAYERS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
# Identical across arms, varying per game. Per-game so two games cannot land on the same
# sampler state; identical across arms because the arms must differ ONLY in the treatment.
os.environ["CARNOT_ARC_GENERATOR_SEED"] = os.environ["CELL_SAMPLER_SEED"]
for _k, _v in ARM_ENV[ARM].items():
    os.environ[_k] = _v
# Positively clear the knobs the treatment arm must NOT have set, rather than trusting the
# launching environment to be clean. An inherited CARNOT_ARC_INDUCE_REPEAT_PENALTY would turn
# `trt` into a second control silently.
for _k in ("CARNOT_ARC_INDUCE_REPEAT_PENALTY", "CARNOT_ARC_INDUCE_DEFECT_REASKS"):
    if _k not in ARM_ENV[ARM]:
        os.environ.pop(_k, None)

sys.path.insert(0, os.path.join(MAIN_REPO, "python"))

# The held-out identity handed to the POLICY while the ENV keeps running the real game -- same
# formula as the retention grid and the 07-30 probe, so all three are comparable.
ANON = "hg" + hashlib.sha256(f"{GAME}|heldout".encode()).hexdigest()[:6]

_T0 = time.monotonic()
RECORDED: list[str] = []
PRE: dict = {}


def _write(payload: dict) -> None:
    tmp = OUT + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, OUT)


def _on_sigterm(signum, frame):  # noqa: ANN001
    """A probe killed by the driver's hard cap must still leave a record.

    An invisible gap would silently unbalance the arms, and this pre-flight's whole point is
    that a missing observation is never a zero. PRE is merged first so every provenance stamp
    gathered before the kill survives instead of having to be inferred afterwards.
    """
    _write({
        **PRE,
        "arm": ARM, "game": GAME, "seed": SEED,
        "status": "blocked_hard_timeout_sigterm",
        "elapsed_s_at_kill": round(time.monotonic() - _T0, 1),
        "result": {"action_trace": list(RECORDED), "timed_out": True},
    })
    os._exit(124)


signal.signal(signal.SIGTERM, _on_sigterm)


def _label(kind, data) -> str:  # noqa: ANN001
    """Canonical encoding of one decided action.

    Ints are normalized because numpy scalars repr differently from plain ints and would fake
    a divergence between two runs that in fact chose the same action.
    """
    if kind == "RESET":
        return "RESET"
    if kind is None:
        return "NONE"
    if isinstance(data, dict):
        d = {k: (int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v)
             for k, v in sorted(data.items())}
        return f"ACTION{kind}|{json.dumps(d, sort_keys=True)}"
    return f"ACTION{kind}|{data!r}"


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", "-C", MAIN_REPO, *args], capture_output=True, text=True,
                             timeout=15)
        return (out.stdout or out.stderr).strip()
    except Exception as exc:
        return f"<error {type(exc).__name__}>"[:120]


def _agentic_content_sha() -> str:
    """Content hash of the live-path package. Two cells sharing this value ran the same code
    whatever their commit shas say -- the stamp that actually identifies what was tested."""
    base = os.path.join(MAIN_REPO, "python", "carnot", "agentic")
    h = hashlib.sha256()
    pairs = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if fn.endswith(".py"):
                p = os.path.join(dirpath, fn)
                with open(p, "rb") as fh:
                    pairs.append((os.path.relpath(p, base), hashlib.sha256(fh.read()).digest()))
    for name, blob in sorted(pairs):
        h.update(name.encode())
        h.update(blob)
    return h.hexdigest()[:16]


def main() -> int:
    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_competition_agent as aca
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_frame_change_predictor as fcp

    global PRE
    pre = PRE
    pre.update({
        "arm": ARM, "game": GAME, "seed": SEED, "anon_game_id": ANON,
        "arm_repo": MAIN_REPO,
        "gpu_requested": GPU, "budget": BUDGET, "max_inductions": MAX_IND,
        "wall_s_cap": WALL_S, "explore_budget": EXPLORE_BUDGET,
        "arm_env_declared": dict(ARM_ENV[ARM]),
        # READ BACK through the shipped accessors, not echoed off os.environ: a cell must never
        # be able to claim a treatment setting the code did not actually apply. This is the
        # single most load-bearing stamp in the file -- it is what proves the arms differ.
        "induce_repeat_penalty_effective": e3._induce_repeat_penalty(),
        "induce_defect_reasks_effective": e3._induce_defect_reasks(),
        "generator_sampler_seed_env": os.environ.get("CARNOT_ARC_GENERATOR_SEED"),
        "generator_sampler_seed_effective": e3.LocalGGUFProposer.sampling_seed(0),
        "arm_commit_measured": _git("rev-parse", "HEAD"),
        "arm_git_status_porcelain": _git("status", "--porcelain")[:2000],
        "agentic_content_sha": _agentic_content_sha(),
        "trace_instrument": "native_action_trace_same_commit_both_arms",
    })

    # The assets whose silent absence turned four cells of the 07-30 grid into a false
    # positive. Here both arms read the SAME canonical checkout so they cannot differ -- but
    # "cannot differ" is exactly the kind of claim that rots, so it is stamped, not argued.
    pre["assets_present"] = {
        p: os.path.exists(os.path.join(MAIN_REPO, p))
        for p in ("results/experiment_4629_live_frame_change_cnn.pt", "data/arc_transition_corpus")
    }
    try:
        _sc = fcp.load_live_action_effect_scorer(root=aca.REPO)
        pre["live_action_effect_scorer"] = None if _sc is None else type(_sc).__name__
    except Exception as exc:
        pre["live_action_effect_scorer"] = f"<error {type(exc).__name__}: {exc}>"[:200]

    pre["e3_module_file"] = e3.__file__
    if not pre["e3_module_file"].startswith(MAIN_REPO):
        _write({**pre, "status": "blocked_wrong_code_imported"})
        return 8

    resolved = str(getattr(e3, "E3_DIR", ""))
    pre["e3_dir_resolved"] = resolved
    if os.path.abspath(resolved) != os.path.abspath(E3_DIR):
        _write({**pre, "status": "blocked_e3_dir_not_honoured"})
        return 2
    pre["e3_dir_entries_at_start"] = sorted(os.listdir(E3_DIR))
    if pre["e3_dir_entries_at_start"]:
        _write({**pre, "status": "blocked_e3_dir_not_empty_at_start"})
        return 7

    # ---- THE INSTRUMENT ------------------------------------------------------------------
    # Both arms are the same commit, so the NATIVE ProgressResult.action_trace is already one
    # identical instrument. The wrapper exists only to force the anonymized policy game id
    # (so the policy cannot route on per-game prior knowledge) and to keep a second,
    # independently-computed trace as a cross-check on the shipped one.
    _BasePolicy = aca.E3AgentPolicy

    class _TracingPolicy(_BasePolicy):  # type: ignore[misc,valid-type]
        def __init__(self, game_id, *a, **kw):  # noqa: ANN001,ANN002,ANN003
            super().__init__(ANON, *a, **kw)

        def next_move(self, frames, latest):  # noqa: ANN001
            kind, data = super().next_move(frames, latest)
            RECORDED.append(_label(kind, data))
            return kind, data

    aca.E3AgentPolicy = _TracingPolicy
    pre["instrument_patched"] = True

    ARM_NAME = "frozen_gemma_pin"
    pre["arm_config_name"] = ARM_NAME
    pre["arm_config_effective"] = dict(atp.ARM_CONFIGS[ARM_NAME])

    prop = e3.LocalGGUFProposer(
        repo_substr=REPO_SUBSTR, port=PORT, mtp=False, n_ctx=32768,
        ffn_cpu_layers=0, kv_quant="q8_0",
    )
    ok = prop._ensure_server()
    pre["server_started"] = bool(ok)
    if not ok:
        _write({**pre, "status": "blocked_generator_server_not_started",
                "selection_log": list(e3.GENERATOR_SELECTION_LOG)[-25:]})
        return 3
    try:
        binary = prop._proc.args[0] if getattr(prop, "_proc", None) else "reused_existing"
    except Exception:
        binary = "unknown"
    pre["server_binary"] = binary
    pre["observed_model_path"] = prop.observed_model_path()
    pre["observed_n_ctx"] = prop.observed_n_ctx()
    pre["server_port"] = prop.port
    if isinstance(binary, str) and "build-hip" in binary:
        _write({**pre, "status": "blocked_generator_bound_igpu_hip_build"})
        return 4
    if pre["observed_model_path"] and REPO_SUBSTR not in str(pre["observed_model_path"]):
        _write({**pre, "status": "blocked_generator_wrong_model"})
        return 5
    try:
        pid_out = subprocess.run(["ss", "-lptnH", f"sport = :{PORT}"], capture_output=True,
                                 text=True, timeout=5).stdout
        m = re.search(r"pid=(\d+)", pid_out)
        spid = int(m.group(1)) if m else None
        pre["server_pid"] = spid
        smi = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_bus_id",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=5).stdout
        pre["vram_rows_mine"] = [r.strip() for r in smi.splitlines()
                                 if spid is not None and r.strip().startswith(str(spid))]
        if spid is not None:
            # The CUDA-vs-HIP question answered by reading the bound binary, not by trusting
            # the launch flags. This is the check that catches an LLM-OFF run reporting LLM-ON.
            pre["server_exe"] = os.path.realpath(f"/proc/{spid}/exe")
    except Exception as exc:
        pre["vram_probe_error"] = repr(exc)[:200]

    res = atp.run_bounded_progress(GAME, ARM_NAME, proposer=prop, seed=SEED,
                                   budget=BUDGET, max_inductions=MAX_IND, wall_s=WALL_S,
                                   explore_budget=EXPLORE_BUDGET, policy_game_id=ANON)
    row = res.to_row(include_events=True, include_trace=True)
    pre["native_action_trace"] = row.pop("action_trace", None)
    row["action_trace"] = list(RECORDED)
    # Did the treatment's own tier actually FIRE? Separates "the tier did not help" from "the
    # tier never fired", which the two are routinely confused and which no trace diff can tell
    # apart on its own.
    try:
        pre["liveness_witness"] = prop.liveness_witness()
    except Exception as exc:
        pre["liveness_witness"] = f"<error {type(exc).__name__}>"[:120]
    pre["n_induce_defect_reasks_observed"] = int(getattr(prop, "n_induce_defect_reasks", -1))

    _write({**pre, "status": "ok", "result": row,
            "cell_wall_s": round(time.monotonic() - _T0, 1)})
    print(json.dumps({"arm": ARM, "game": GAME, "actions": row.get("total_actions"),
                      "trace_len": len(RECORDED), "timed_out": row.get("timed_out"),
                      "n_ind": row.get("n_inductions"),
                      "reasks": pre["n_induce_defect_reasks_observed"],
                      "wall": round(time.monotonic() - _T0, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # a crash is a datum, not a silent gap
        _write({**PRE, "arm": ARM, "game": GAME, "seed": SEED,
                "status": "blocked_cell_exception",
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "result": {"action_trace": list(RECORDED), "timed_out": False},
                "cell_wall_s": round(time.monotonic() - _T0, 1)})
        raise
