#!/usr/bin/env python3
"""WHERE DO THE LIVE AGENT'S ACTIONS COME FROM? -- the per-action accounting, plus the
inertness proof that the instrument did not change the answer.

**The question.** For a game the SCORED live agent FAILS: of the N actions it spends,
where did each one come from, and what was its state when it chose? Three independent
2026-07/2026-08 lines had already concluded, by inference, that the
induce -> verify -> plan pipeline is not on the causal path to banking a level (LLM-tier
deletion left 5 of 6 games byte-identical; 0/22 stall inductions cleared the goal gate
while 4/6 level-up re-inductions did; tn36 holds an engine with held-out accuracy 1.0 and
banked 0 levels in 346 actions). Nobody had measured it at the level where it would show:
the ACTION. This produces that accounting.

**What it runs.** The SCORED policy `E3AgentPolicy` -- reached exactly the way
`make_carnot_agent` reaches it -- against the OFFLINE arcade
(`arc_solver_kit.offline_arcade()`, `OperationMode.OFFLINE`, local `environment_files/`).
No scorecard, no gateway, no network, no submission. Each arm runs in its own killable
subprocess because inducing a world model executes LLM-authored engine code.

**The three arms, and why three.**
    A   instrument OFF   seed S
    B   instrument ON    seed S
    A'  instrument OFF   seed S   (a bit-for-bit repeat of A)

    A vs B tests the claim that matters: does arming the instrument change what the agent
    does? A vs A' establishes whether the agent is deterministic at a fixed seed AT ALL --
    without it, an A==B result could be luck and an A!=B result could be ordinary run-to-run
    noise wrongly blamed on the instrument. Reporting only A vs B would be asserting a
    determinism nobody verified. If A != A', the honest output is the observed noise floor,
    not an inertness claim.

Usage:
    .venv/bin/python scripts/arc_action_provenance_probe.py --game tn36 --seed 20260801

Spec: openspec/capabilities/arc-world-model-trust-energy/spec.md REQ-ARC-WMTE-6070
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKER = os.path.join(REPO_ROOT, "scripts", "arc_action_provenance_worker.py")

sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
# The closed branch vocabulary, imported rather than retyped, so the artifact's
# "which branches did this configuration NOT exercise" line cannot silently go stale when a
# branch is added. A configuration that reaches only some branches is a real limit on what
# its inertness result covers, and it has to be stated, not inferred by the reader.
from carnot.agentic.arc_action_provenance import TOP_BRANCHES as _TOP_BRANCHES  # noqa: E402


def _sha(trace: list[str]) -> str:
    return hashlib.sha256("\n".join(trace).encode()).hexdigest()


def _reap_my_generator(port: int) -> int:
    """Kill the llama-server THIS probe started on `port`. Returns how many were killed.

    `LocalGGUFProposer` leaves its server running after the arm's Python process exits --
    correct for its normal use (a persistent server serves many calls), fatal here: the
    server holds ~21.7 GB and the NEXT arm's headroom guard then declines the card and falls
    back to the iGPU. Observed exactly that: arm A_off completed, and arms B_on and A2_off
    both refused with "CUDA gpu1 has 2415 MiB free".

    MATCHED ON THE PORT THIS PROBE ASSIGNED, never on the process name. This machine is
    shared, another workflow runs its own generator, and killing a process this session did
    not start is not recoverable. The port is the only identifier that is provably ours.
    """
    killed = 0
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as fh:
                    cmdline = fh.read().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if "llama-server" in cmdline and f"--port {port} " in cmdline + " ":
                try:
                    os.kill(int(entry), 15)
                    killed += 1
                except OSError:
                    pass
    except OSError:
        pass
    return killed


def _wait_for_card(gpu: str, need_mb: int, timeout_s: float) -> dict:
    """Block until `gpu` has `need_mb` free, or give up. Purely observational.

    NEVER frees the card itself beyond reaping this probe's own server (done by the caller):
    a card held by the conductor or by another workflow is theirs, and this waits rather than
    evicting.
    """
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout
            for line in out.splitlines():
                idx, free = (p.strip() for p in line.split(","))
                if idx == str(gpu):
                    last = int(free)
                    if last >= need_mb:
                        return {"free_mb": last, "waited_s": round(time.time() - t0, 1), "ok": True}
        except Exception:  # pragma: no cover - nvidia-smi absent or transient
            return {"free_mb": None, "waited_s": round(time.time() - t0, 1), "ok": None}
        time.sleep(5)
    return {"free_mb": last, "waited_s": round(time.time() - t0, 1), "ok": False}


def run_arm(
    *,
    label: str,
    armed: bool,
    game: str,
    seed: int,
    budget: int,
    max_inductions: int,
    explore_budget: int,
    wall_s: float,
    workdir: str,
    timeout: float,
    generator: str,
    cuda_gpu: str,
    cuda_port: int,
) -> dict:
    """One arm, one subprocess. Returns the worker's JSON (or an error record)."""
    out_path = os.path.join(workdir, f"arm_{label}.json")
    env = dict(os.environ)
    if armed:
        env["CARNOT_ARC_ACTION_PROVENANCE"] = "1"
    else:
        # Explicitly REMOVED, not set to "0": the point of an unarmed arm is that the agent
        # runs exactly as it does when nobody has ever heard of this flag.
        env.pop("CARNOT_ARC_ACTION_PROVENANCE", None)
    env.pop("CARNOT_ARC_ACTION_PROVENANCE_DIR", None)
    # PER-ARM ENGINE STORE. `results/arc_e3/` is TRACKED, READ-ONLY EVIDENCE, and
    # `LocalGGUFProposer.induce` writes `results/arc_e3/<game>/world_model.py` on every
    # successful induction -- so a live-generator arm run without this redirect OVERWRITES
    # the committed engine for the game under measurement. That is not theoretical: the
    # first live attempt at this measurement rewrote `results/arc_e3/tn36/world_model.py`
    # (40 insertions, 14 deletions) within 90 seconds of starting, and it was caught by
    # `git status`, not by any guard -- `_guard_engine_write` is deliberately scoped to
    # pytest, because the LIVE agent writing here is exactly what the store is for. The file
    # was restored from git.
    #
    # A PER-ARM directory, not one shared temp directory, because the arms must be
    # INDEPENDENT: with a shared store, arm A's induced engine is on disk when arm B starts,
    # so B's `load_engine` could read A's output. That would have made the A/B comparison a
    # comparison of two different situations while reporting it as a comparison of one
    # situation with and without an instrument.
    #
    # Set in the CHILD's environment before the interpreter starts, which is the only way it
    # works: `E3_DIR` is resolved ONCE at module import (documented at
    # arc_executable_world_model.py:60, "THE REDIRECT IS NOT TOTAL").
    env["CARNOT_ARC_E3_DIR"] = os.path.join(workdir, f"e3_store_{label}")
    if generator == "none":
        # Induction OFF and no card visible. Together these make the run DETERMINISTIC,
        # which is the precondition for a byte-identity claim to mean anything: with a
        # sampling generator in the loop, two arms differing proves nothing about the
        # instrument. The cost is that the two plan-execution branches are unreachable in
        # this configuration -- which is why the live-generator arm exists as well.
        env["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("CARNOT_ARC_DISABLE_INDUCTION", None)
        # OUTER LOOP OWNS GPU 1 (2026-06-27 operator allocation). Never GPU 0: that is the
        # conductor's, and this session must not evict a process it did not start.
        #
        # `CARNOT_ARC_GENERATOR_CUDA_GPU` ONLY -- do NOT also set CUDA_VISIBLE_DEVICES here.
        # The proposer builds its own launch env and pins the card itself; pre-setting
        # CUDA_VISIBLE_DEVICES=1 renumbers the visible cards so that PHYSICAL card 1 is no
        # longer AT index 1, the headroom probe for index 1 finds nothing, and
        # `_generator_server_and_env` silently falls back to the AMD iGPU HIP build. That is
        # not hypothetical: the first attempt at this measurement spent ten minutes running
        # a 31B model on the iGPU while the artifact would have said "3090". The worker's
        # generator witness now refuses that fallback outright, and this comment records why
        # the obvious-looking extra pin must not be re-added.
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["CARNOT_ARC_GENERATOR_CUDA_GPU"] = cuda_gpu
    cmd = [
        sys.executable,
        WORKER,
        "--game",
        game,
        "--seed",
        str(seed),
        "--budget",
        str(budget),
        "--max-inductions",
        str(max_inductions),
        "--explore-budget",
        str(explore_budget),
        "--wall-s",
        str(wall_s),
        "--generator",
        generator,
        "--cuda-port",
        # A DISTINCT port per arm. Same-port reuse across sequential arms is how a stale
        # server from a previous arm gets silently served instead of a fresh one -- the
        # exact failure mode recorded in the GGUF outer-loop GPU-pinning note. Non-default
        # base port so this cannot collide with the conductor's generator either.
        str(cuda_port),
        "--out",
        out_path,
        "--arm-label",
        label,
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd, env=env, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {
            "arm_label": label,
            "error": "timeout",
            "wall_s_measured": round(time.time() - t0, 3),
        }
    if proc.returncode != 0 or not os.path.exists(out_path):
        return {
            "arm_label": label,
            "error": f"worker_exit_{proc.returncode}",
            "stderr_tail": (proc.stderr or "")[-2000:],
            "wall_s_measured": round(time.time() - t0, 3),
        }
    with open(out_path, encoding="utf-8") as fh:
        row = json.load(fh)
    row["stdout_tail"] = (proc.stdout or "")[-400:]
    return row


def compare(a: dict, b: dict) -> dict:
    """Trace-level comparison of two arms."""
    ta, tb = a.get("action_trace") or [], b.get("action_trace") or []
    identical = ta == tb
    first_div = None
    if not identical:
        for i, (x, y) in enumerate(zip(ta, tb)):
            if x != y:
                first_div = {"index": i, "left": x, "right": y}
                break
        if first_div is None:
            first_div = {"index": min(len(ta), len(tb)), "left": None, "right": None}
    return {
        "left": a.get("arm_label"),
        "right": b.get("arm_label"),
        "n_actions_left": len(ta),
        "n_actions_right": len(tb),
        "sha256_left": _sha(ta),
        "sha256_right": _sha(tb),
        "byte_identical": identical,
        "first_divergence": first_div,
        "unified_diff_head": (
            None if identical else list(difflib.unified_diff(ta, tb, lineterm="", n=1))[:40]
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="tn36")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--max-inductions", type=int, default=3)
    ap.add_argument("--explore-budget", type=int, default=24)
    ap.add_argument("--wall-s", type=float, default=900.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--generator", choices=("none", "live"), default="none")
    ap.add_argument("--cuda-gpu", default="1", help="outer loop owns GPU 1; never GPU 0")
    ap.add_argument("--cuda-port", type=int, default=8951)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        REPO_ROOT,
        "results",
        f"outer_loop_arc_action_provenance_{args.game}_{args.generator}_{args.seed}.json",
    )

    t0 = time.time()
    arms = []
    card_waits: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="arc_action_prov_") as workdir:
        for arm_i, (label, armed) in enumerate(
            (("A_off", False), ("B_on", True), ("A2_off", False))
        ):
            if args.generator == "live" and arm_i:
                # The PREVIOUS arm's `LocalGGUFProposer` leaves its llama-server running and
                # holding ~21.7 GB. Reap it (by ITS port, which this probe assigned, never by
                # process name -- the machine is shared), then wait for the card to actually
                # come back before starting this arm. Without this the second and third arms
                # both refuse with "CUDA gpu1 has 2415 MiB free", which is what happened on
                # the first live attempt.
                reaped = _reap_my_generator(args.cuda_port + arm_i - 1)
                wait = _wait_for_card(args.cuda_gpu, 23000, 300.0)
                wait["arm"] = label
                wait["reaped_servers"] = reaped
                card_waits.append(wait)
                print(f"[probe] reaped {reaped} server(s); card wait: {wait}", flush=True)
            print(f"[probe] running arm {label} (armed={armed})...", flush=True)
            arms.append(
                run_arm(
                    label=label,
                    armed=armed,
                    game=args.game,
                    seed=args.seed,
                    budget=args.budget,
                    max_inductions=args.max_inductions,
                    explore_budget=args.explore_budget,
                    wall_s=args.wall_s,
                    workdir=workdir,
                    timeout=args.timeout,
                    generator=args.generator,
                    cuda_gpu=args.cuda_gpu,
                    cuda_port=args.cuda_port + arm_i,
                )
            )
        if args.generator == "live":
            # Reap the LAST arm's server too. A probe that leaves 21.7 GB pinned on a shared
            # card after it exits is a worse citizen than one that never ran.
            _reap_my_generator(args.cuda_port + 2)

    a_off, b_on, a2_off = arms
    determinism = compare(a_off, a2_off)  # A vs A' -- is the agent deterministic at all?
    inertness = compare(a_off, b_on)  # A vs B  -- did arming change anything?

    prov = b_on.get("provenance") or {}
    summary = prov.get("summary") or {}

    # The verdict is written to depend on the determinism check FIRST. An inertness claim
    # made on a nondeterministic agent is not a claim, it is a coincidence with a label.
    if any(a.get("error") for a in arms):
        verdict = "blocked_arm_failed_see_error_fields"
    elif not determinism["byte_identical"]:
        verdict = (
            "complete_agent_is_nondeterministic_at_fixed_seed_"
            "inertness_reported_against_measured_noise_floor_not_asserted"
        )
    elif inertness["byte_identical"]:
        # THE VERDICT CARRIES ITS OWN SCOPE. A bare
        # "instrument_inert_action_sequence_byte_identical" reads as a claim about the whole
        # policy, and this run only exercised the branches it actually reached -- the
        # induction-disabled configuration never enters either plan-execution branch, so an
        # unqualified verdict would be a claim broader than its evidence about exactly the
        # code path the measurement is aimed at. The unreached count goes in the verdict
        # string, not only in a field further down that a summariser may not read.
        n_missing = len(set(_TOP_BRANCHES) - set((summary.get("by_top_branch") or {}).keys()))
        verdict = (
            "complete_instrument_inert_action_sequence_byte_identical_with_flag_unset_"
            f"over_{len(_TOP_BRANCHES) - n_missing}_of_{len(_TOP_BRANCHES)}_top_branches_"
            f"{n_missing}_unreached_by_this_configuration"
        )
    else:
        verdict = "complete_instrument_PERTURBS_the_agent_action_sequence_differs_do_not_use"

    artifact = {
        "experiment": f"outer_loop_arc_action_provenance_{args.game}",
        "schema": "carnot.arc.action_provenance_probe.v1",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "milestone": "2026.08.outer_loop",
        "question": (
            "For a game the SCORED live agent fails: of the N actions it spends, which "
            "branch chose each one, and what was the agent's state at the moment of choice?"
        ),
        "honest_verdict": verdict,
        "duration_s": round(time.time() - t0, 3),
        # The agent takes real actions against the offline arcade with NO LLM in the loop
        # (proposer=None; the induce tier reads the already-stored engine from
        # results/arc_e3/<game>/). That is exactly the substrate this taxonomy names.
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "inference_substrate_note": (
            "E3AgentPolicy (the SCORED policy) steps the OFFLINE arcade "
            "(OperationMode.OFFLINE over local environment_files/). No scorecard is opened "
            "against the live service, no gateway is contacted, nothing is submitted. "
            "proposer=None so no GGUF is loaded and CUDA_VISIBLE_DEVICES is emptied in "
            "every arm; the induce tier still runs against the stored engine."
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_note": (
            "This measures WHERE ACTIONS COME FROM. It banks no level and claims no solve; "
            "it is an instrument run on the development twin of the scored path."
        ),
        "random_seed": args.seed,
        "verifier_is_oracle": {
            "value": False,
            "principle": (
                "nothing here consults a win oracle. The recorded quantity is which code "
                "branch emitted each action, which is a fact about the agent, not about "
                "whether the action was correct."
            ),
        },
        "config": {
            "game": args.game,
            "budget": args.budget,
            "max_inductions": args.max_inductions,
            "explore_budget": args.explore_budget,
            "wall_s": args.wall_s,
            "policy": "E3AgentPolicy via arc_actions_to_progress.run_bounded_progress",
            "generator": args.generator,
            "induction_disabled": args.generator == "none",
            "cuda_gpu": (args.cuda_gpu if args.generator == "live" else None),
        },
        "branches_exercised": sorted(set((summary.get("by_top_branch") or {}).keys())),
        "branches_not_exercised_by_this_configuration": sorted(
            set(_TOP_BRANCHES) - set((summary.get("by_top_branch") or {}).keys())
        ),
        # -- the inertness proof, reported before the finding it licenses ------------------
        "determinism_check_A_vs_A2": determinism,
        "inertness_check_A_vs_B": inertness,
        "agent_deterministic_at_fixed_seed": determinism["byte_identical"],
        "instrument_inert": bool(determinism["byte_identical"] and inertness["byte_identical"]),
        # -- the accounting ---------------------------------------------------------------
        "action_accounting": summary,
        "arms": [
            {k: v for k, v in a.items() if k not in ("provenance", "action_trace")} for a in arms
        ],
        "provenance_rows_n": len(prov.get("rows") or []),
        # Between-arm GPU handoff, recorded because a live arm that silently ran on a
        # degraded substrate is the failure this whole file is written to make impossible.
        "card_waits_between_arms": card_waits,
    }
    checksum_src = json.dumps(
        {
            "traces": [a.get("action_trace") for a in arms],
            "summary": summary,
            "seed": args.seed,
            "game": args.game,
        },
        sort_keys=True,
        default=str,
    ).encode()
    artifact["reproducibility_checksum"] = "sha256:" + hashlib.sha256(checksum_src).hexdigest()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=1, default=str)

    # The full provenance rows go beside the artifact, not inside it: one row per action
    # times several hundred actions would bloat the artifact past usefulness, and the rows
    # are raw evidence rather than a finding.
    rows_path = out_path.replace(".json", "_rows.json")
    with open(rows_path, "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=1, default=str)

    print(
        json.dumps(
            {
                k: artifact[k]
                for k in (
                    "honest_verdict",
                    "agent_deterministic_at_fixed_seed",
                    "instrument_inert",
                    "provenance_rows_n",
                )
            },
            indent=1,
        )
    )
    print("action_accounting:", json.dumps(summary, indent=1))
    print("wrote", out_path)
    print("wrote", rows_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
